"""
Feast Feature Store Definitions.

This module defines the feature views and entities for the customer churn
prediction system using Feast feature store.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Feature Store                               │
    │  ┌─────────────────────────────────────────────────────────┐   │
    │  │                    Data Sources                          │   │
    │  │  ┌─────────────────┐      ┌─────────────────────────┐   │   │
    │  │  │  churn_stats    │      │  customer_demographics  │   │   │
    │  │  │  (Parquet)      │      │      (Parquet)          │   │   │
    │  │  └────────┬────────┘      └────────┬────────────────┘   │   │
    │  └───────────┼────────────────────────┼─────────────────────┘   │
    │              │                        │                         │
    │  ┌───────────▼────────────────────────▼─────────────────────┐   │
    │  │                    Feature Views                          │   │
    │  │  ┌─────────────────┐      ┌─────────────────────────┐    │   │
    │  │  │churn_stats_view │      │customer_demographics_fv │    │   │
    │  │  │  (19 features)  │      │     (6 features)        │    │   │
    │  │  └────────┬────────┘      └────────┬────────────────┘    │   │
    │  └───────────┼────────────────────────┼─────────────────────┘   │
    │              │                        │                         │
    │              └────────────┬───────────┘                         │
    │                           │                                     │
    │              ┌────────────▼────────────┐                        │
    │              │    Feature Service      │                        │
    │              │   churn_prediction_fs   │                        │
    │              └─────────────────────────┘                        │
    └─────────────────────────────────────────────────────────────────┘

Feature Categories:
    1. Behavioral Features: Usage patterns, support interactions
    2. Financial Features: Charges, payment history
    3. Contractual Features: Contract type, tenure
    4. Demographic Features: Age, location, family status

Example:
    >>> from feast import FeatureStore
    >>> store = FeatureStore(repo_path="feature_repo/")
    >>> features = store.get_online_features(
    ...     features=["churn_stats_view:tenure", "churn_stats_view:monthly_charges"],
    ...     entity_rows=[{"customer_id": "CUST001"}]
    ... )
    >>> print(features.to_dict())

Author: Shadow MLOps Team
Version: 2.0.0
"""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float32, Float64, Int64, String

# =============================================================================
# Entities
# =============================================================================

customer = Entity(
    name="customer_id",
    description="Unique customer identifier for churn prediction",
    join_keys=["customer_id"],
    tags={
        "team": "ml-platform",
        "pii": "false",
    },
)

# =============================================================================
# Data Sources
# =============================================================================

# Primary churn statistics data source
churn_stats_source = FileSource(
    name="churn_stats_source",
    path="data/churn_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
    description="Customer behavioral and transactional features for churn prediction",
    tags={
        "source_system": "crm",
        "update_frequency": "daily",
    },
)

# Customer demographics data source
customer_demographics_source = FileSource(
    name="customer_demographics_source",
    path="data/customer_demographics.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
    description="Customer demographic and profile information",
    tags={
        "source_system": "customer_master",
        "update_frequency": "weekly",
    },
)

# =============================================================================
# Feature Views
# =============================================================================

# Churn Statistics Feature View
# Contains behavioral, financial, and service-related features
churn_stats_view = FeatureView(
    name="churn_stats_view",
    description="Customer churn prediction features including behavioral and financial signals",
    entities=[customer],
    ttl=timedelta(days=90),  # Feature freshness window
    schema=[
        # Tenure and Contract Features
        Field(
            name="tenure",
            dtype=Float32,
            description="Months as customer (0-72)",
            tags={"category": "contract"},
        ),
        Field(
            name="contract_type",
            dtype=Int64,
            description="Contract type: 0=Month-to-month, 1=One-year, 2=Two-year",
            tags={"category": "contract"},
        ),
        # Financial Features
        Field(
            name="monthly_charges",
            dtype=Float32,
            description="Monthly billing amount in USD",
            tags={"category": "financial"},
        ),
        Field(
            name="total_charges",
            dtype=Float32,
            description="Total charges to date in USD",
            tags={"category": "financial"},
        ),
        Field(
            name="payment_method",
            dtype=Int64,
            description="Payment method: 0=Credit card, 1=Bank transfer, 2=Electronic check, 3=Mailed check",
            tags={"category": "financial"},
        ),
        Field(
            name="paperless_billing",
            dtype=Int64,
            description="Uses paperless billing: 0=No, 1=Yes",
            tags={"category": "financial"},
        ),
        # Service Features
        Field(
            name="phone_service",
            dtype=Int64,
            description="Has phone service: 0=No, 1=Yes",
            tags={"category": "service"},
        ),
        Field(
            name="multiple_lines",
            dtype=Int64,
            description="Has multiple lines: 0=No, 1=Yes",
            tags={"category": "service"},
        ),
        Field(
            name="internet_service",
            dtype=Int64,
            description="Internet service type: 0=None, 1=DSL, 2=Fiber optic",
            tags={"category": "service"},
        ),
        Field(
            name="online_security",
            dtype=Int64,
            description="Has online security: 0=No, 1=Yes",
            tags={"category": "service"},
        ),
        Field(
            name="online_backup",
            dtype=Int64,
            description="Has online backup: 0=No, 1=Yes",
            tags={"category": "service"},
        ),
        Field(
            name="device_protection",
            dtype=Int64,
            description="Has device protection: 0=No, 1=Yes",
            tags={"category": "service"},
        ),
        Field(
            name="tech_support",
            dtype=Int64,
            description="Has tech support: 0=No, 1=Yes",
            tags={"category": "service"},
        ),
        Field(
            name="streaming_tv",
            dtype=Int64,
            description="Has streaming TV: 0=No, 1=Yes",
            tags={"category": "service"},
        ),
        Field(
            name="streaming_movies",
            dtype=Int64,
            description="Has streaming movies: 0=No, 1=Yes",
            tags={"category": "service"},
        ),
        # Behavioral Features
        Field(
            name="num_support_tickets",
            dtype=Int64,
            description="Number of support tickets in last 90 days",
            tags={"category": "behavioral"},
        ),
        Field(
            name="avg_monthly_gb_download",
            dtype=Float32,
            description="Average monthly data download in GB",
            tags={"category": "behavioral"},
        ),
        Field(
            name="num_referrals",
            dtype=Int64,
            description="Number of customer referrals made",
            tags={"category": "behavioral"},
        ),
        Field(
            name="satisfaction_score",
            dtype=Float32,
            description="Latest CSAT score (1-5 scale)",
            tags={"category": "behavioral"},
        ),
    ],
    source=churn_stats_source,
    online=True,
    tags={
        "team": "ml-platform",
        "model": "churn_prediction",
        "version": "2.0",
    },
)

# Customer Demographics Feature View
# Contains demographic and profile information
customer_demographics_view = FeatureView(
    name="customer_demographics_view",
    description="Customer demographic features for segmentation and personalization",
    entities=[customer],
    ttl=timedelta(days=365),  # Demographics change less frequently
    schema=[
        Field(
            name="gender",
            dtype=String,
            description="Customer gender: Male, Female, Other",
            tags={"category": "demographic", "pii": "true"},
        ),
        Field(
            name="senior_citizen",
            dtype=Int64,
            description="Is senior citizen (65+): 0=No, 1=Yes",
            tags={"category": "demographic"},
        ),
        Field(
            name="partner",
            dtype=Int64,
            description="Has partner: 0=No, 1=Yes",
            tags={"category": "demographic"},
        ),
        Field(
            name="num_dependents",
            dtype=Int64,
            description="Number of dependents",
            tags={"category": "demographic"},
        ),
        Field(
            name="location_cluster",
            dtype=Int64,
            description="Geographic location cluster ID",
            tags={"category": "demographic"},
        ),
        Field(
            name="customer_segment",
            dtype=String,
            description="Customer segment: premium, standard, basic",
            tags={"category": "demographic"},
        ),
    ],
    source=customer_demographics_source,
    online=True,
    tags={
        "team": "ml-platform",
        "model": "churn_prediction",
        "version": "2.0",
    },
)

# =============================================================================
# Feature Services
# =============================================================================

# Churn Prediction Feature Service
# Bundles all features needed for the churn prediction model
churn_prediction_service = FeatureService(
    name="churn_prediction_service",
    description="All features required for churn prediction model inference",
    features=[
        churn_stats_view,
        customer_demographics_view,
    ],
    tags={
        "team": "ml-platform",
        "model": "churn_prediction",
        "environment": "production",
    },
)

# Lightweight Feature Service
# For low-latency predictions with essential features only
churn_prediction_lite_service = FeatureService(
    name="churn_prediction_lite_service",
    description="Essential features for low-latency churn predictions",
    features=[
        churn_stats_view[["tenure", "monthly_charges", "contract_type", "num_support_tickets"]],
    ],
    tags={
        "team": "ml-platform",
        "model": "churn_prediction",
        "environment": "production",
        "latency": "low",
    },
)
