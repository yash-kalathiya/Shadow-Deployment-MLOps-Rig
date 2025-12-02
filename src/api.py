"""
Shadow Deployment API for Churn Prediction

This FastAPI application implements a shadow deployment pattern where:
- Champion Model: Production model that serves predictions to users
- Challenger Model: Shadow model that runs in parallel for comparison

The Challenger model's predictions are logged asynchronously for analysis
without affecting production response times or user experience.

Author: MLOps Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import aiofiles
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.exceptions import (
    CircuitBreakerOpenError,
    ModelNotLoadedError,
    ModelPredictionError,
    RateLimitExceededError,
    ShadowMLOpsError,
)
from src.models import ChallengerModel, ChampionModel

# =============================================================================
# CONFIGURATION
# =============================================================================


class Config:
    """Application configuration with sensible defaults."""
    
    APP_NAME: str = "Shadow-MLOps API"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Paths
    LOG_DIR: Path = Path("logs")
    SHADOW_LOG_FILE: Path = LOG_DIR / "shadow_logs.json"
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Shadow deployment
    SHADOW_MODE_ENABLED: bool = True
    SHADOW_LOG_MAX_ENTRIES: int = 10000
    
    # Circuit breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 30  # seconds


config = Config()

# Ensure log directory exists
config.LOG_DIR.mkdir(parents=True, exist_ok=True)


# Custom logger with request context
class RequestContextFilter(logging.Filter):
    """Add request_id to log records for distributed tracing."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "system"
        return True


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / "api.log"),
    ],
)

logger = logging.getLogger("shadow_api")
logger.addFilter(RequestContextFilter())


# =============================================================================
# ENUMS
# =============================================================================


class RiskCategory(str, Enum):
    """Churn risk categories with clear thresholds."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelRole(str, Enum):
    """Model roles in shadow deployment."""
    
    CHAMPION = "champion"
    CHALLENGER = "challenger"


# =============================================================================
# REQUEST/RESPONSE MODELS (Pydantic v2)
# =============================================================================


class CustomerFeatures(BaseModel):
    """
    Input features for churn prediction.
    
    All features are validated with realistic bounds to catch input errors early.
    """
    
    customer_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique customer identifier",
        json_schema_extra={"example": "CUST_001234"},
    )
    days_since_last_login: int = Field(
        ...,
        ge=0,
        le=3650,
        description="Days since last login (max 10 years)",
        json_schema_extra={"example": 7},
    )
    login_frequency_30d: float = Field(
        ...,
        ge=0,
        le=100,
        description="Login frequency in last 30 days",
        json_schema_extra={"example": 12.5},
    )
    session_duration_avg: float = Field(
        ...,
        ge=0,
        le=1440,
        description="Average session duration in minutes (max 24h)",
        json_schema_extra={"example": 25.0},
    )
    total_transactions_90d: int = Field(
        ...,
        ge=0,
        le=10000,
        description="Total transactions in last 90 days",
        json_schema_extra={"example": 8},
    )
    transaction_value_avg: float = Field(
        ...,
        ge=0,
        le=1000000,
        description="Average transaction value",
        json_schema_extra={"example": 150.0},
    )
    support_tickets_30d: int = Field(
        ...,
        ge=0,
        le=1000,
        description="Support tickets in last 30 days",
        json_schema_extra={"example": 1},
    )
    subscription_tenure_days: int = Field(
        ...,
        ge=0,
        le=36500,
        description="Days since subscription started (max 100 years)",
        json_schema_extra={"example": 365},
    )
    satisfaction_score: float = Field(
        ...,
        ge=0,
        le=10,
        description="Customer satisfaction score (0-10)",
        json_schema_extra={"example": 7.5},
    )
    
    @field_validator("customer_id")
    @classmethod
    def normalize_customer_id(cls, v: str) -> str:
        """Normalize customer ID to uppercase."""
        return v.strip().upper()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "customer_id": "CUST_001234",
                    "days_since_last_login": 7,
                    "login_frequency_30d": 12.5,
                    "session_duration_avg": 25.0,
                    "total_transactions_90d": 8,
                    "transaction_value_avg": 150.0,
                    "support_tickets_30d": 1,
                    "subscription_tenure_days": 365,
                    "satisfaction_score": 7.5,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Prediction response with full traceability."""
    
    request_id: str = Field(..., description="Unique request ID for tracing")
    customer_id: str = Field(..., description="Customer identifier")
    churn_probability: float = Field(..., ge=0, le=1, description="Churn probability (0-1)")
    churn_prediction: bool = Field(..., description="Binary churn prediction")
    risk_category: RiskCategory = Field(..., description="Risk category")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    model_version: str = Field(..., description="Model version")
    model_name: str = Field(..., description="Model name")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    latency_ms: float = Field(..., description="Prediction latency in ms")


class HealthResponse(BaseModel):
    """Comprehensive health check response."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")
    models: Dict[str, Dict[str, Any]] = Field(..., description="Model status")
    dependencies: Dict[str, str] = Field(..., description="Dependency health")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request with size limits."""
    
    customers: List[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of customers (max 100)",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response with statistics."""
    
    request_id: str = Field(..., description="Batch request ID")
    predictions: List[PredictionResponse] = Field(..., description="Predictions")
    total_count: int = Field(..., description="Total count")
    success_count: int = Field(..., description="Successful predictions")
    failed_count: int = Field(..., description="Failed predictions")
    processing_time_ms: float = Field(..., description="Total processing time in ms")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    
    error: str = Field(..., description="Error type")
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    request_id: Optional[str] = Field(None, description="Request ID")
    timestamp: str = Field(..., description="Error timestamp")


# =============================================================================
# RATE LIMITER (Sliding Window Algorithm)
# =============================================================================


class RateLimiter:
    """
    Thread-safe sliding window rate limiter.
    
    For production with multiple instances, use Redis-based implementation.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining/retry_after)."""
        async with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove expired entries
            self._requests[client_id] = [
                ts for ts in self._requests[client_id] if ts > window_start
            ]
            
            if len(self._requests[client_id]) >= self.max_requests:
                oldest = min(self._requests[client_id])
                retry_after = int(oldest + self.window_seconds - now) + 1
                return False, retry_after
            
            self._requests[client_id].append(now)
            remaining = self.max_requests - len(self._requests[client_id])
            return True, remaining


# =============================================================================
# CIRCUIT BREAKER (Fault Tolerance)
# =============================================================================


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    States: CLOSED (normal) -> OPEN (failing) -> HALF_OPEN (testing)
    """
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 30):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self._failures: Dict[str, int] = defaultdict(int)
        self._last_failure: Dict[str, float] = {}
        self._state: Dict[str, str] = defaultdict(lambda: "CLOSED")
        self._lock = asyncio.Lock()
    
    async def is_open(self, service: str) -> bool:
        """Check if circuit is open (rejecting requests)."""
        async with self._lock:
            if self._state[service] == "OPEN":
                if time.time() - self._last_failure.get(service, 0) > self.timeout_seconds:
                    self._state[service] = "HALF_OPEN"
                    return False
                return True
            return False
    
    async def record_success(self, service: str) -> None:
        """Record successful call, reset circuit."""
        async with self._lock:
            self._failures[service] = 0
            self._state[service] = "CLOSED"
    
    async def record_failure(self, service: str) -> None:
        """Record failure, potentially open circuit."""
        async with self._lock:
            self._failures[service] += 1
            self._last_failure[service] = time.time()
            if self._failures[service] >= self.failure_threshold:
                self._state[service] = "OPEN"
    
    def get_retry_after(self, service: str) -> int:
        """Get seconds until retry is allowed."""
        elapsed = time.time() - self._last_failure.get(service, 0)
        return max(1, int(self.timeout_seconds - elapsed))


# =============================================================================
# SHADOW LOGGER (Async with Buffering)
# =============================================================================


class ShadowLogger:
    """
    High-performance async shadow logger with buffering.
    
    Buffers entries and flushes periodically to minimize I/O overhead.
    """
    
    def __init__(self, log_file: Path, max_entries: int = 10000):
        self.log_file = log_file
        self.max_entries = max_entries
        self._lock = asyncio.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._buffer_size = 50
        self._ensure_file()
    
    def _ensure_file(self) -> None:
        """Ensure log file exists."""
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
        """Log shadow prediction comparison."""
        entry = {
            "request_id": request_id,
            "customer_id": customer_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "champion": {
                "prediction": champion_prediction,
                "latency_ms": round(latency_champion_ms, 3),
                "model_version": champion_prediction.get("model_version"),
            },
            "challenger": {
                "prediction": challenger_prediction,
                "latency_ms": round(latency_challenger_ms, 3),
                "model_version": challenger_prediction.get("model_version"),
            },
            "comparison": {
                "probability_diff": round(
                    abs(
                        champion_prediction["churn_probability"]
                        - challenger_prediction["churn_probability"]
                    ),
                    4,
                ),
                "prediction_match": (
                    champion_prediction["churn_prediction"]
                    == challenger_prediction["churn_prediction"]
                ),
            },
        }
        
        async with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._buffer_size:
                await self._flush()
    
    async def _flush(self) -> None:
        """Flush buffer to file."""
        if not self._buffer:
            return
        
        try:
            async with aiofiles.open(self.log_file, "r") as f:
                content = await f.read()
                logs = json.loads(content) if content.strip() else []
            
            logs.extend(self._buffer)
            if len(logs) > self.max_entries:
                logs = logs[-self.max_entries:]
            
            async with aiofiles.open(self.log_file, "w") as f:
                await f.write(json.dumps(logs, indent=2))
            
            self._buffer.clear()
        except Exception as e:
            logger.error(f"Shadow log flush failed: {e}", extra={"request_id": "system"})
    
    async def force_flush(self) -> None:
        """Force flush all buffered entries."""
        async with self._lock:
            await self._flush()


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

rate_limiter = RateLimiter(config.RATE_LIMIT_REQUESTS, config.RATE_LIMIT_WINDOW)
circuit_breaker = CircuitBreaker(config.CIRCUIT_BREAKER_THRESHOLD, config.CIRCUIT_BREAKER_TIMEOUT)
shadow_logger: Optional[ShadowLogger] = None
champion_model: Optional[ChampionModel] = None
challenger_model: Optional[ChallengerModel] = None


# =============================================================================
# DEPENDENCIES
# =============================================================================


async def get_client_id(request: Request) -> str:
    """Extract client ID from request for rate limiting."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def check_rate_limit(request: Request, client_id: str = Depends(get_client_id)) -> None:
    """Rate limit dependency."""
    if not config.RATE_LIMIT_ENABLED:
        return
    
    allowed, value = await rate_limiter.is_allowed(client_id)
    if not allowed:
        raise RateLimitExceededError(retry_after=value)
    request.state.rate_limit_remaining = value


def get_request_id(request: Request) -> str:
    """Get or generate request ID for tracing."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global shadow_logger, champion_model, challenger_model
    
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}", extra={"request_id": "startup"})
    
    # Load models
    champion_model = ChampionModel()
    challenger_model = ChallengerModel()
    logger.info(
        f"Models: Champion v{champion_model.version}, Challenger v{challenger_model.version}",
        extra={"request_id": "startup"},
    )
    
    # Initialize shadow logger
    shadow_logger = ShadowLogger(config.SHADOW_LOG_FILE, config.SHADOW_LOG_MAX_ENTRIES)
    
    logger.info("Ready to serve requests", extra={"request_id": "startup"})
    
    yield
    
    # Shutdown
    logger.info("Shutting down...", extra={"request_id": "shutdown"})
    if shadow_logger:
        await shadow_logger.force_flush()
    logger.info("Shutdown complete", extra={"request_id": "shutdown"})


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Shadow-MLOps: Churn Prediction API",
    description="""
## 🚀 Enterprise Shadow Deployment for ML Models

This API implements a production-grade shadow deployment pattern for churn prediction.

### Architecture
| Component | Description |
|-----------|-------------|
| **Champion Model** | Production model serving real predictions |
| **Challenger Model** | Shadow model for comparison (async) |

### Features
- ✅ Zero-downtime model comparison
- ✅ Async shadow prediction logging  
- ✅ Rate limiting (100 req/min)
- ✅ Circuit breaker pattern
- ✅ Request tracing (X-Request-ID)
- ✅ Prometheus-compatible metrics

### Rate Limits
Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests
    """,
    version=config.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "MLOps Team", "email": "mlops@example.com"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Limit", "X-Process-Time-Ms"],
)


# =============================================================================
# MIDDLEWARE
# =============================================================================


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request tracing and timing."""
    request_id = get_request_id(request)
    request.state.request_id = request_id
    
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    
    if hasattr(request.state, "rate_limit_remaining"):
        response.headers["X-RateLimit-Remaining"] = str(request.state.rate_limit_remaining)
        response.headers["X-RateLimit-Limit"] = str(config.RATE_LIMIT_REQUESTS)
    
    logger.info(
        f"{request.method} {request.url.path} {response.status_code} {elapsed:.1f}ms",
        extra={"request_id": request_id},
    )
    return response


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================


@app.exception_handler(ShadowMLOpsError)
async def handle_app_error(request: Request, exc: ShadowMLOpsError):
    """Handle application-specific errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """Handle unexpected errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unexpected error: {exc}", extra={"request_id": request_id}, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def classify_risk(probability: float) -> RiskCategory:
    """Classify risk based on churn probability."""
    if probability < 0.25:
        return RiskCategory.LOW
    elif probability < 0.50:
        return RiskCategory.MEDIUM
    elif probability < 0.75:
        return RiskCategory.HIGH
    return RiskCategory.CRITICAL


async def run_shadow_prediction(
    request_id: str,
    customer_id: str,
    features: Dict[str, Any],
    champion_result: Dict[str, Any],
    champion_latency: float,
) -> None:
    """Run challenger prediction asynchronously."""
    if not config.SHADOW_MODE_ENABLED or not challenger_model:
        return
    
    start = time.perf_counter()
    try:
        result = challenger_model.predict(features)
        latency = (time.perf_counter() - start) * 1000
        
        await shadow_logger.log_shadow_prediction(
            request_id, customer_id, champion_result, result, features, champion_latency, latency
        )
        await circuit_breaker.record_success("challenger")
    except Exception as e:
        await circuit_breaker.record_failure("challenger")
        logger.error(f"Shadow prediction failed: {e}", extra={"request_id": request_id})


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/", include_in_schema=False)
async def root():
    """Root redirect to docs."""
    return {"service": config.APP_NAME, "version": config.APP_VERSION, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=config.APP_VERSION,
        environment=config.ENVIRONMENT,
        models={
            "champion": {
                "version": f"v{champion_model.version}" if champion_model else "unavailable",
                "name": champion_model.model_name if champion_model else "unknown",
                "status": "healthy" if champion_model else "unavailable",
            },
            "challenger": {
                "version": f"v{challenger_model.version}" if challenger_model else "unavailable",
                "name": challenger_model.model_name if challenger_model else "unknown",
                "status": "healthy" if challenger_model else "unavailable",
            },
        },
        dependencies={
            "shadow_logger": "healthy" if shadow_logger else "unavailable",
            "rate_limiter": "healthy",
            "circuit_breaker": "healthy",
        },
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(
    request: CustomerFeatures,
    background_tasks: BackgroundTasks,
    req: Request,
    _: None = Depends(check_rate_limit),
) -> PredictionResponse:
    """
    Predict customer churn probability.
    
    Returns Champion model result. Challenger runs async in shadow mode.
    """
    request_id = getattr(req.state, "request_id", str(uuid.uuid4()))
    features = request.model_dump()
    
    if not champion_model:
        raise ModelNotLoadedError("champion")
    
    if await circuit_breaker.is_open("champion"):
        raise CircuitBreakerOpenError("champion", circuit_breaker.get_retry_after("champion"))
    
    start = time.perf_counter()
    try:
        result = champion_model.predict(features)
        latency = (time.perf_counter() - start) * 1000
        await circuit_breaker.record_success("champion")
    except Exception as e:
        await circuit_breaker.record_failure("champion")
        raise ModelPredictionError("champion", str(e))
    
    # Shadow prediction (async)
    if challenger_model:
        background_tasks.add_task(
            run_shadow_prediction, request_id, request.customer_id, features, result, latency
        )
    
    return PredictionResponse(
        request_id=request_id,
        customer_id=request.customer_id,
        churn_probability=result["churn_probability"],
        churn_prediction=result["churn_prediction"],
        risk_category=classify_risk(result["churn_probability"]),
        confidence=result.get("confidence", 0.0),
        model_version=f"v{champion_model.version}",
        model_name=champion_model.model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_ms=round(latency, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    _: None = Depends(check_rate_limit),
) -> BatchPredictionResponse:
    """Batch prediction for multiple customers (max 100)."""
    batch_id = getattr(req.state, "request_id", str(uuid.uuid4()))
    start = time.perf_counter()
    
    predictions, failed = [], 0
    
    for customer in request.customers:
        try:
            pred_start = time.perf_counter()
            result = champion_model.predict(customer.model_dump())
            latency = (time.perf_counter() - pred_start) * 1000
            
            predictions.append(PredictionResponse(
                request_id=str(uuid.uuid4()),
                customer_id=customer.customer_id,
                churn_probability=result["churn_probability"],
                churn_prediction=result["churn_prediction"],
                risk_category=classify_risk(result["churn_probability"]),
                confidence=result.get("confidence", 0.0),
                model_version=f"v{champion_model.version}",
                model_name=champion_model.model_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                latency_ms=round(latency, 3),
            ))
        except Exception:
            failed += 1
    
    return BatchPredictionResponse(
        request_id=batch_id,
        predictions=predictions,
        total_count=len(request.customers),
        success_count=len(predictions),
        failed_count=failed,
        processing_time_ms=round((time.perf_counter() - start) * 1000, 3),
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus-compatible metrics."""
    try:
        if config.SHADOW_LOG_FILE.exists():
            async with aiofiles.open(config.SHADOW_LOG_FILE, "r") as f:
                logs = json.loads(await f.read() or "[]")
        else:
            logs = []
    except Exception:
        logs = []
    
    total = len(logs)
    matches = sum(1 for l in logs if l.get("comparison", {}).get("prediction_match", False))
    
    return {
        "shadow_metrics": {
            "total_predictions": total,
            "matching_predictions": matches,
            "match_rate": round(matches / total, 4) if total else 0,
            "champion_version": champion_model.version if champion_model else "unknown",
            "challenger_version": challenger_model.version if challenger_model else "unknown",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/shadow/logs", tags=["Shadow Deployment"])
async def get_shadow_logs(
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Get shadow prediction logs."""
    try:
        if config.SHADOW_LOG_FILE.exists():
            async with aiofiles.open(config.SHADOW_LOG_FILE, "r") as f:
                logs = json.loads(await f.read() or "[]")
            return {"logs": logs[-limit:], "total": len(logs)}
        return {"logs": [], "total": 0}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/shadow/comparison", tags=["Shadow Deployment"])
async def get_shadow_comparison():
    """Get model comparison summary."""
    try:
        if not config.SHADOW_LOG_FILE.exists():
            return {"status": "no_data", "message": "No logs yet"}
        
        async with aiofiles.open(config.SHADOW_LOG_FILE, "r") as f:
            logs = json.loads(await f.read() or "[]")
        
        if not logs:
            return {"status": "no_data"}
        
        total = len(logs)
        matches = sum(1 for l in logs if l["comparison"]["prediction_match"])
        diffs = [l["comparison"]["probability_diff"] for l in logs]
        rate = matches / total
        
        return {
            "summary": {
                "total": total,
                "agreement_rate": round(rate, 4),
                "agreements": matches,
                "disagreements": total - matches,
            },
            "probability_diff": {
                "mean": round(sum(diffs) / total, 4),
                "max": round(max(diffs), 4),
                "min": round(min(diffs), 4),
            },
            "recommendation": "PROMOTE" if rate > 0.95 else "MONITOR",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
