"""
Shadow Deployment API for Churn Prediction

This FastAPI application implements a shadow deployment pattern where:
- Champion Model: Production model that serves predictions to users
- Challenger Model: Shadow model that runs in parallel for comparison

The Challenger model's predictions are logged asynchronously for analysis
without affecting production response times or user experience.
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.models import ChampionModel, ChallengerModel

# =============================================================================
# CONFIGURATION
# =============================================================================

LOG_DIR = Path("logs")
SHADOW_LOG_FILE = LOG_DIR / "shadow_logs.json"
PREDICTION_LOG_FILE = LOG_DIR / "prediction_logs.json"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log"),
    ],
)
logger = logging.getLogger("shadow_api")

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class CustomerFeatures(BaseModel):
    """Input features for churn prediction."""

    customer_id: str = Field(..., description="Unique customer identifier")
    days_since_last_login: int = Field(..., ge=0, description="Days since last login")
    login_frequency_30d: float = Field(..., ge=0, description="Login frequency in last 30 days")
    session_duration_avg: float = Field(..., ge=0, description="Average session duration in minutes")
    total_transactions_90d: int = Field(..., ge=0, description="Total transactions in last 90 days")
    transaction_value_avg: float = Field(..., ge=0, description="Average transaction value")
    support_tickets_30d: int = Field(..., ge=0, description="Support tickets in last 30 days")
    subscription_tenure_days: int = Field(..., ge=0, description="Days since subscription started")
    satisfaction_score: float = Field(..., ge=0, le=10, description="Customer satisfaction score")


class PredictionResponse(BaseModel):
    """Response from the prediction endpoint."""

    request_id: str = Field(..., description="Unique request identifier")
    customer_id: str = Field(..., description="Customer identifier")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    churn_prediction: bool = Field(..., description="Binary churn prediction")
    risk_category: str = Field(..., description="Risk category: low, medium, high")
    model_version: str = Field(..., description="Champion model version")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    models: Dict[str, str]
    version: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    request_id: str
    predictions: List[PredictionResponse]
    processing_time_ms: float


# =============================================================================
# SHADOW LOGGING
# =============================================================================


class ShadowLogger:
    """Asynchronous logger for shadow model predictions."""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self._lock = asyncio.Lock()
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Ensure log directory and file exist."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                json.dump([], f)

    async def log_shadow_prediction(
        self,
        request_id: str,
        customer_id: str,
        champion_prediction: Dict[str, Any],
        challenger_prediction: Dict[str, Any],
        features: Dict[str, Any],
        latency_champion_ms: float,
        latency_challenger_ms: float,
    ) -> None:
        """
        Asynchronously log shadow predictions for comparison analysis.

        Args:
            request_id: Unique request identifier
            customer_id: Customer identifier
            champion_prediction: Champion model prediction result
            challenger_prediction: Challenger model prediction result
            features: Input features used for prediction
            latency_champion_ms: Champion model latency in milliseconds
            latency_challenger_ms: Challenger model latency in milliseconds
        """
        log_entry = {
            "request_id": request_id,
            "customer_id": customer_id,
            "timestamp": datetime.utcnow().isoformat(),
            "champion": {
                "prediction": champion_prediction,
                "latency_ms": latency_champion_ms,
                "model_version": champion_prediction.get("model_version", "unknown"),
            },
            "challenger": {
                "prediction": challenger_prediction,
                "latency_ms": latency_challenger_ms,
                "model_version": challenger_prediction.get("model_version", "unknown"),
            },
            "comparison": {
                "probability_diff": abs(
                    champion_prediction["churn_probability"]
                    - challenger_prediction["churn_probability"]
                ),
                "prediction_match": champion_prediction["churn_prediction"]
                == challenger_prediction["churn_prediction"],
            },
            "features": features,
        }

        async with self._lock:
            try:
                async with aiofiles.open(self.log_file, "r") as f:
                    content = await f.read()
                    logs = json.loads(content) if content else []

                logs.append(log_entry)

                # Keep only last 10000 entries to prevent unbounded growth
                if len(logs) > 10000:
                    logs = logs[-10000:]

                async with aiofiles.open(self.log_file, "w") as f:
                    await f.write(json.dumps(logs, indent=2))

                logger.info(
                    f"Shadow log recorded: request_id={request_id}, "
                    f"prediction_match={log_entry['comparison']['prediction_match']}, "
                    f"prob_diff={log_entry['comparison']['probability_diff']:.4f}"
                )

            except Exception as e:
                logger.error(f"Failed to write shadow log: {e}")


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

# Global instances
shadow_logger: Optional[ShadowLogger] = None
champion_model: Optional[ChampionModel] = None
challenger_model: Optional[ChallengerModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global shadow_logger, champion_model, challenger_model

    # Startup
    logger.info("Initializing Shadow Deployment API...")

    # Initialize models
    champion_model = ChampionModel()
    challenger_model = ChallengerModel()
    logger.info(f"Champion Model loaded: v{champion_model.version}")
    logger.info(f"Challenger Model loaded: v{challenger_model.version}")

    # Initialize shadow logger
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    shadow_logger = ShadowLogger(SHADOW_LOG_FILE)
    logger.info(f"Shadow logger initialized: {SHADOW_LOG_FILE}")

    logger.info("Shadow Deployment API ready to serve requests")

    yield

    # Shutdown
    logger.info("Shutting down Shadow Deployment API...")
    champion_model = None
    challenger_model = None
    shadow_logger = None
    logger.info("Shutdown complete")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Shadow-MLOps: Churn Prediction API",
    description="""
    ## Shadow Deployment Pattern for Churn Prediction

    This API implements a shadow deployment architecture where:
    - **Champion Model**: The production model serving real predictions
    - **Challenger Model**: A shadow model running in parallel for comparison

    ### Key Features
    - Zero-downtime model comparison
    - Asynchronous shadow prediction logging
    - Real-time drift detection readiness
    - Feast Feature Store integration

    ### Endpoints
    - `/predict`: Single customer churn prediction
    - `/predict/batch`: Batch predictions for multiple customers
    - `/health`: System health check
    - `/metrics`: Prometheus-compatible metrics
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def classify_risk(probability: float) -> str:
    """Classify churn risk based on probability threshold."""
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    else:
        return "high"


async def run_shadow_prediction(
    request_id: str,
    customer_id: str,
    features: Dict[str, Any],
    champion_result: Dict[str, Any],
    champion_latency: float,
) -> None:
    """
    Run challenger model prediction asynchronously in shadow mode.

    This function runs in the background after the champion prediction
    is returned to the user, ensuring zero latency impact.
    """
    import time

    start_time = time.perf_counter()

    try:
        # Run challenger prediction
        challenger_result = challenger_model.predict(features)
        challenger_latency = (time.perf_counter() - start_time) * 1000

        # Log shadow comparison
        await shadow_logger.log_shadow_prediction(
            request_id=request_id,
            customer_id=customer_id,
            champion_prediction=champion_result,
            challenger_prediction=challenger_result,
            features=features,
            latency_champion_ms=champion_latency,
            latency_challenger_ms=challenger_latency,
        )

    except Exception as e:
        logger.error(f"Shadow prediction failed for request {request_id}: {e}")


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to documentation."""
    return {"message": "Shadow-MLOps API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for load balancer and monitoring.

    Returns the status of all system components including model versions.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        models={
            "champion": f"v{champion_model.version}" if champion_model else "not_loaded",
            "challenger": f"v{challenger_model.version}" if challenger_model else "not_loaded",
        },
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: CustomerFeatures,
    background_tasks: BackgroundTasks,
    req: Request,
) -> PredictionResponse:
    """
    Predict customer churn probability.

    This endpoint:
    1. Runs the Champion (production) model and returns the result
    2. Asynchronously runs the Challenger (shadow) model
    3. Logs both predictions for comparison analysis

    The user only sees the Champion model result, ensuring consistent
    production behavior while enabling shadow comparison.

    Args:
        request: Customer features for prediction

    Returns:
        PredictionResponse: Champion model's churn prediction
    """
    import time

    request_id = str(uuid.uuid4())
    features = request.model_dump()

    # Validate models are loaded
    if not champion_model or not challenger_model:
        raise HTTPException(
            status_code=503,
            detail="Models not initialized. Please retry later.",
        )

    # Run Champion model (production)
    start_time = time.perf_counter()
    try:
        champion_result = champion_model.predict(features)
        champion_latency = (time.perf_counter() - start_time) * 1000
    except Exception as e:
        logger.error(f"Champion model prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please retry later.",
        )

    # Schedule Challenger model to run asynchronously (shadow mode)
    background_tasks.add_task(
        run_shadow_prediction,
        request_id,
        request.customer_id,
        features,
        champion_result,
        champion_latency,
    )

    # Return ONLY the Champion result to the user
    response = PredictionResponse(
        request_id=request_id,
        customer_id=request.customer_id,
        churn_probability=champion_result["churn_probability"],
        churn_prediction=champion_result["churn_prediction"],
        risk_category=classify_risk(champion_result["churn_probability"]),
        model_version=f"v{champion_model.version}",
        timestamp=datetime.utcnow().isoformat(),
    )

    logger.info(
        f"Prediction served: request_id={request_id}, "
        f"customer_id={request.customer_id}, "
        f"churn_prob={champion_result['churn_probability']:.4f}"
    )

    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
) -> BatchPredictionResponse:
    """
    Batch prediction endpoint for multiple customers.

    Processes multiple customers in a single request while maintaining
    shadow deployment logging for each individual prediction.

    Args:
        request: List of customer features

    Returns:
        BatchPredictionResponse: Predictions for all customers
    """
    import time

    batch_request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    predictions = []

    for customer in request.customers:
        request_id = str(uuid.uuid4())
        features = customer.model_dump()

        # Champion prediction
        pred_start = time.perf_counter()
        champion_result = champion_model.predict(features)
        champion_latency = (time.perf_counter() - pred_start) * 1000

        # Schedule shadow prediction
        background_tasks.add_task(
            run_shadow_prediction,
            request_id,
            customer.customer_id,
            features,
            champion_result,
            champion_latency,
        )

        predictions.append(
            PredictionResponse(
                request_id=request_id,
                customer_id=customer.customer_id,
                churn_probability=champion_result["churn_probability"],
                churn_prediction=champion_result["churn_prediction"],
                risk_category=classify_risk(champion_result["churn_probability"]),
                model_version=f"v{champion_model.version}",
                timestamp=datetime.utcnow().isoformat(),
            )
        )

    processing_time = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"Batch prediction served: batch_id={batch_request_id}, "
        f"count={len(predictions)}, "
        f"processing_time_ms={processing_time:.2f}"
    )

    return BatchPredictionResponse(
        request_id=batch_request_id,
        predictions=predictions,
        processing_time_ms=processing_time,
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Returns metrics for monitoring shadow deployment performance.
    """
    # Read shadow logs for metrics
    try:
        if SHADOW_LOG_FILE.exists():
            with open(SHADOW_LOG_FILE, "r") as f:
                logs = json.load(f)
        else:
            logs = []
    except Exception:
        logs = []

    total_predictions = len(logs)
    matching_predictions = sum(
        1 for log in logs if log.get("comparison", {}).get("prediction_match", False)
    )
    avg_prob_diff = (
        sum(log.get("comparison", {}).get("probability_diff", 0) for log in logs) / total_predictions
        if total_predictions > 0
        else 0
    )

    return {
        "shadow_deployment_metrics": {
            "total_shadow_predictions": total_predictions,
            "matching_predictions": matching_predictions,
            "match_rate": matching_predictions / total_predictions if total_predictions > 0 else 0,
            "average_probability_difference": avg_prob_diff,
            "champion_model_version": champion_model.version if champion_model else "unknown",
            "challenger_model_version": challenger_model.version if challenger_model else "unknown",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/shadow/logs", tags=["Shadow Deployment"])
async def get_shadow_logs(limit: int = 100):
    """
    Retrieve recent shadow deployment logs for analysis.

    Args:
        limit: Maximum number of log entries to return (default: 100)

    Returns:
        Recent shadow prediction logs
    """
    try:
        if SHADOW_LOG_FILE.exists():
            with open(SHADOW_LOG_FILE, "r") as f:
                logs = json.load(f)
            return {"logs": logs[-limit:], "total_count": len(logs)}
        return {"logs": [], "total_count": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")


@app.get("/shadow/comparison", tags=["Shadow Deployment"])
async def get_shadow_comparison():
    """
    Get comparison summary between Champion and Challenger models.

    Returns aggregated statistics for shadow deployment analysis.
    """
    try:
        if not SHADOW_LOG_FILE.exists():
            return {
                "message": "No shadow logs available yet",
                "recommendation": "Send predictions to generate comparison data",
            }

        with open(SHADOW_LOG_FILE, "r") as f:
            logs = json.load(f)

        if not logs:
            return {"message": "No shadow logs available yet"}

        # Calculate statistics
        total = len(logs)
        matches = sum(1 for log in logs if log["comparison"]["prediction_match"])
        prob_diffs = [log["comparison"]["probability_diff"] for log in logs]

        champion_latencies = [log["champion"]["latency_ms"] for log in logs]
        challenger_latencies = [log["challenger"]["latency_ms"] for log in logs]

        return {
            "summary": {
                "total_comparisons": total,
                "prediction_agreement_rate": matches / total,
                "prediction_disagreements": total - matches,
            },
            "probability_difference": {
                "mean": sum(prob_diffs) / total,
                "max": max(prob_diffs),
                "min": min(prob_diffs),
            },
            "latency_comparison": {
                "champion_avg_ms": sum(champion_latencies) / total,
                "challenger_avg_ms": sum(challenger_latencies) / total,
            },
            "recommendation": (
                "Challenger ready for promotion"
                if matches / total > 0.95
                else "Continue monitoring - prediction agreement below threshold"
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate comparison: {e}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
