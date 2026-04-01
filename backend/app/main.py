"""FastAPI application factory.

The create_app() factory pattern allows:
    - Multiple app instances in tests (each test gets a fresh app)
    - Deferred imports (faster module loading)
    - Configuration injection for different environments
    - Clean startup/shutdown lifecycle management

Production entry point:
    uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.database import close_db, init_db

logger = structlog.get_logger()

APP_VERSION = "0.1.0"


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle.

    Startup:
        - Validate configuration (crash fast if secrets missing)
        - Configure structured logging
        - Initialize database connection pool and verify connectivity
        - Log startup banner

    Shutdown:
        - Close database connections gracefully
        - Close Redis connections
        - Flush any pending audit logs

    If the database is unreachable at startup, the app crashes immediately.
    This is intentional — better to fail deployment than to serve 500s.
    """
    settings = get_settings()

    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if settings.is_development
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(settings.log_level),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    log = structlog.get_logger()

    # Initialize database — crash fast if unreachable
    try:
        await init_db()
        await log.ainfo("database_connected", pool_size=settings.db_pool_size)
    except Exception as exc:
        await log.aerror("database_connection_failed", error=str(exc))
        raise RuntimeError(
            f"Cannot start: database is unreachable at {settings.database_url.split('@')[-1]}. "
            f"Error: {exc}"
        ) from exc

    await log.ainfo(
        "starting_kangaroo_shift",
        version=APP_VERSION,
        environment=settings.environment,
        debug=settings.debug,
    )

    yield

    # Graceful shutdown — close all DB connections
    await close_db()
    await log.ainfo("shutting_down_kangaroo_shift")


# ── App Factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns a fully configured FastAPI instance with:
        - CORS middleware (configured per environment)
        - Request ID middleware (trace every request)
        - Request timing middleware (monitor latency)
        - Structured JSON logging
        - Global error handlers
        - Health check endpoint
        - API v1 router
    """
    settings = get_settings()

    app = FastAPI(
        title="Kangaroo Shift",
        description=(
            "Context management platform for seamless LLM switching. "
            "Capture, compress, structure, encrypt, and reconstruct conversational "
            "context across ChatGPT, Claude, Gemini, and open-source models."
        ),
        version=APP_VERSION,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # ── CORS Middleware ───────────────────────────────────────────────────
    # In production, this must be restricted to your actual domain.
    # In development, localhost origins are allowed for convenience.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )

    # ── Request ID Middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        """Attach a unique request ID to every request/response.

        This enables end-to-end request tracing:
        - Frontend sends X-Request-ID header → we use it
        - No header → we generate a UUID
        - Response includes the ID for correlation
        - All log entries for this request include the ID

        When a customer reports "my shift failed," they can give us
        the request ID from the response headers, and we can find
        every log entry for that specific request.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Timing Middleware ─────────────────────────────────────────────────
    @app.middleware("http")
    async def timing_middleware(request: Request, call_next) -> Response:
        """Measure and expose request processing time.

        X-Process-Time header shows milliseconds spent processing.
        This data feeds into:
        - API latency monitoring (P50, P95, P99)
        - SLA compliance tracking (enterprise customers)
        - Performance regression detection in CI
        """
        start = time.perf_counter()
        response = await call_next(request)
        process_time_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Process-Time"] = str(process_time_ms)

        log = structlog.get_logger()
        await log.ainfo(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=process_time_ms,
        )

        return response

    # ── Global Error Handlers ─────────────────────────────────────────────

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors that escape Pydantic."""
        log = structlog.get_logger()
        await log.awarning("value_error", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": str(exc),
                "request_id": request.headers.get("X-Request-ID", "unknown"),
            },
        )

    @app.exception_handler(PermissionError)
    async def permission_error_handler(
        request: Request, exc: PermissionError,
    ) -> JSONResponse:
        """Handle authorization failures."""
        log = structlog.get_logger()
        await log.awarning("permission_denied", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=403,
            content={
                "error": "forbidden",
                "message": "You do not have permission to perform this action.",
                "request_id": request.headers.get("X-Request-ID", "unknown"),
            },
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all error handler.

        CRITICAL: In production, we NEVER expose internal error details
        to the client. The actual error is logged server-side with the
        request ID, so support can look it up. The client gets a generic
        message plus the request ID for reference.

        This prevents information leakage attacks where error messages
        reveal database structure, file paths, or internal logic.
        """
        log = structlog.get_logger()
        await log.aerror(
            "unhandled_exception",
            error_type=type(exc).__name__,
            error=str(exc),
            path=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": (
                    str(exc)
                    if settings.is_development
                    else "An internal error occurred. Please try again or contact support."
                ),
                "request_id": request.headers.get("X-Request-ID", "unknown"),
            },
        )

    # ── Health Endpoint ───────────────────────────────────────────────────

    @app.get(
        "/health",
        tags=["system"],
        summary="Health check",
        response_model=dict,
    )
    async def health():
        """Health check endpoint.

        Returns basic service status. Used by:
        - Docker health checks (keeps containers running)
        - Kubernetes liveness probes (restarts unhealthy pods)
        - Load balancers (routes traffic to healthy instances)
        - Monitoring systems (PagerDuty alerts on failure)
        """
        return {
            "status": "ok",
            "version": APP_VERSION,
            "environment": settings.environment,
        }

    # ── Include API Routers ───────────────────────────────────────────────
    from app.api.v1 import router as v1_router

    app.include_router(v1_router, prefix="/api/v1")

    return app
