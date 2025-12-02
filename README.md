# 🚀 Shadow Deployment & Drift Detection Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/linting-ruff-purple)](https://github.com/astral-sh/ruff)

A production-grade MLOps platform implementing **shadow deployment patterns** for safe model rollouts with comprehensive **statistical drift detection**. Built with modern Python best practices and designed for enterprise scalability.

## 🎯 Key Features

- **Shadow Deployment**: Run Champion and Challenger models simultaneously without affecting production
- **Statistical Drift Detection**: PSI, KS-test, and Jensen-Shannon divergence for data quality monitoring
- **Feature Store Integration**: Feast-based feature management for consistent feature serving
- **Real-time Predictions**: Low-latency FastAPI inference with async support
- **Automated Retraining**: GitHub Actions workflow for drift-triggered model updates
- **Production Ready**: Rate limiting, circuit breakers, structured logging, health checks

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Shadow Deployment Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌──────────────────────────────────────────────┐      │
│   │   Client    │────▶│              FastAPI Gateway                  │      │
│   └─────────────┘     │  • Rate Limiting  • Request Tracing          │      │
│                       │  • Input Validation  • Error Handling         │      │
│                       └───────────────────┬──────────────────────────┘      │
│                                           │                                  │
│                       ┌───────────────────┴───────────────────┐              │
│                       │                                       │              │
│                       ▼                                       ▼              │
│          ┌────────────────────┐               ┌────────────────────┐        │
│          │   Champion Model   │               │  Challenger Model  │        │
│          │      (v2.1.0)      │               │   (v3.0.0-beta)    │        │
│          │  ┌──────────────┐  │               │  ┌──────────────┐  │        │
│          │  │ 89.2% Acc    │  │               │  │ 90.8% Acc    │  │        │
│          │  │ 0.923 AUC    │  │               │  │ 0.941 AUC    │  │        │
│          │  └──────────────┘  │               │  └──────────────┘  │        │
│          └─────────┬──────────┘               └─────────┬──────────┘        │
│                    │                                    │                    │
│          ┌─────────▼──────────┐               ┌─────────▼──────────┐        │
│          │   Return to        │               │   Log to Shadow    │        │
│          │   Client           │               │   Storage          │        │
│          └────────────────────┘               └────────────────────┘        │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                     Drift Detection Pipeline                      │      │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │      │
│   │  │   PSI   │  │KS Test  │  │  J-S    │  │  Automated Retrain  │ │      │
│   │  │Detector │  │Detector │  │Diverge  │  │  Trigger (>0.3 PSI) │ │      │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘ │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                      Feast Feature Store                          │      │
│   │  ┌────────────────────┐    ┌────────────────────────────────┐   │      │
│   │  │ churn_stats_view   │    │ customer_demographics_view     │   │      │
│   │  │ • 19 features      │    │ • 6 features                   │   │      │
│   │  │ • 90-day TTL       │    │ • 365-day TTL                  │   │      │
│   │  └────────────────────┘    └────────────────────────────────┘   │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yash-kalathiya/Shadow-Deployment-MLOps-Rig.git
cd Shadow-Deployment-MLOps-Rig

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install-dev  # or: pip install -r requirements.txt
```

### Running the API

```bash
# Development mode with hot reload
make run

# Production mode
make run-prod

# With Docker
make docker-build
make docker-run
```

### Making Predictions

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST-001",
    "tenure": 24,
    "monthly_charges": 75.50,
    "contract_type": 1,
    "num_support_tickets": 2
  }'

# Response
{
  "probability": 0.42,
  "label": 0,
  "confidence": 0.84,
  "risk_tier": "MEDIUM",
  "model_version": "2.1.0",
  "request_id": "abc123-def456"
}
```

## 📁 Project Structure

```
Shadow-Deployment-MLOps-Rig/
├── 📂 .github/workflows/     # CI/CD pipelines
│   └── retrain.yml           # Automated retraining workflow
├── 📂 feature_repo/          # Feast feature store
│   ├── feature_store.yaml    # Store configuration
│   └── definitions.py        # Feature definitions
├── 📂 src/                   # Main application code
│   ├── api.py                # FastAPI application
│   ├── config.py             # Configuration management
│   ├── exceptions.py         # Custom exceptions
│   └── models.py             # ML model implementations
├── 📂 monitoring/            # Observability
│   └── detect_drift.py       # Drift detection engine
├── 📂 tests/                 # Test suite
│   ├── conftest.py           # Shared fixtures
│   ├── test_api.py           # API tests
│   ├── test_models.py        # Model tests
│   └── test_drift.py         # Drift detection tests
├── 📂 scripts/               # Utility scripts
│   └── train_model.py        # Training script
├── 📄 Dockerfile             # Container definition
├── 📄 docker-compose.yml     # Service orchestration
├── 📄 Makefile               # Development commands
├── 📄 pyproject.toml         # Project configuration
└── 📄 requirements.txt       # Dependencies
```

## 🔬 Shadow Deployment Pattern

Shadow deployment allows safe evaluation of new models by:

1. **Champion serves production traffic** - Users always get predictions from the proven model
2. **Challenger runs in parallel** - New model makes predictions but results are logged, not served
3. **Compare offline** - Analyze challenger performance without production risk
4. **Promote with confidence** - When challenger outperforms, swap with zero downtime

```python
@app.post("/predict")
async def predict(request: PredictionRequest):
    # Champion serves the user
    champion_result = champion_model.predict(request.features)
    
    # Challenger runs in shadow (async, non-blocking)
    asyncio.create_task(
        shadow_log_prediction(challenger_model, request)
    )
    
    return champion_result  # Only champion is returned
```

## 📈 Drift Detection

The platform supports multiple statistical methods:

| Method | Use Case | Threshold |
|--------|----------|-----------|
| **PSI** (Population Stability Index) | Continuous features | 0.3 = significant drift |
| **KS Test** (Kolmogorov-Smirnov) | Distribution comparison | 0.1 = significant difference |
| **Jensen-Shannon Divergence** | Symmetric measure | 0.1 = notable divergence |

### Running Drift Detection

```bash
# Run with sample data
make drift-check

# Generate detailed report
make drift-report

# Custom thresholds
python -m monitoring.detect_drift --psi-threshold 0.2 --generate-sample
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_models.py -v

# Run fast tests only
make test-fast
```

## 📊 Metrics & Monitoring

### Health Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Overall system health |
| `GET /health/ready` | Kubernetes readiness probe |
| `GET /health/live` | Kubernetes liveness probe |
| `GET /models` | Model metadata and statistics |

### Prometheus Metrics

- `predictions_total` - Total predictions by model
- `prediction_latency_seconds` - Prediction latency histogram
- `drift_score` - Current drift score by feature
- `model_prediction_count` - Predictions per model version

## 🔧 Configuration

Configuration is managed through environment variables and Pydantic Settings:

```python
# src/config.py
class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Shadow MLOps API"
    
    # Model Configuration
    model_champion_version: str = "2.1.0"
    model_challenger_version: str = "3.0.0-beta"
    
    # Drift Detection
    drift_psi_threshold: float = 0.3
    drift_check_interval_hours: int = 24
    
    class Config:
        env_file = ".env"
```

## 🚢 Deployment

### Docker

```bash
# Build image
docker build -t shadow-mlops:latest .

# Run container
docker run -p 8000:8000 shadow-mlops:latest
```

### Docker Compose (Full Stack)

```bash
# Start all services (API, Prometheus, Grafana)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Kubernetes (Helm)

```bash
# Coming soon
helm install shadow-mlops ./charts/shadow-mlops
```

## 📚 API Documentation

Once running, access the interactive API docs:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
make install-dev

# Run quality checks
make quality

# Run pre-commit hooks
make pre-commit
```

## 🔐 Security

For security concerns, please see [SECURITY.md](SECURITY.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Feast](https://feast.dev/) - Feature store for ML
- [Evidently AI](https://evidentlyai.com/) - ML monitoring inspiration
- [Pydantic](https://pydantic.dev/) - Data validation

---

<p align="center">
  Built with ❤️ for the MLOps community
</p>
