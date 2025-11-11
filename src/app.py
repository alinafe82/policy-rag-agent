"""FastAPI application for Policy RAG Agent with production-ready features."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .cache import SimpleCache
from .config import get_settings
from .guard import approve_or_refuse
from .llm import get_llm
from .logger import log_error, setup_logger
from .store import InMemoryStore

logger = setup_logger(__name__)
settings = get_settings()

# Global state (initialized in lifespan)
store: InMemoryStore
cache: SimpleCache
request_metrics: dict[str, int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global store, cache

    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")

    store = InMemoryStore.from_samples()
    cache = SimpleCache(
        ttl_seconds=settings.cache_ttl_seconds, max_size=settings.cache_max_size
    )

    request_metrics["total_requests"] = 0
    request_metrics["successful_requests"] = 0
    request_metrics["failed_requests"] = 0

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down application")
    cache.clear()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Safe, auditable RAG system for enterprise policy documents with citation validation",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions gracefully."""
    log_error(
        logger,
        exc,
        {
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else "unknown",
        },
    )

    request_metrics["failed_requests"] += 1

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": (
                str(exc)
                if not settings.is_production
                else "An unexpected error occurred"
            ),
        },
    )


# Pydantic models
class Question(BaseModel):
    """Question request with validation."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User question about policies",
        json_schema_extra={"example": "What are the MFA requirements?"},
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or only whitespace")
        return v


class Answer(BaseModel):
    """Answer response with metadata."""

    answer: str = Field(..., description="Generated answer with citations")
    sources: list[str] = Field(..., description="List of source document IDs")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    citations_found: int = Field(..., description="Number of citations in answer")
    cached: bool = Field(False, description="Whether result was served from cache")
    process_time_ms: float = Field(..., description="Processing time in milliseconds")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    environment: str
    store_size: int
    cache_enabled: bool


class MetricsResponse(BaseModel):
    """Metrics response."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    cache_enabled: bool


# API Endpoints
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["monitoring"],
    summary="Health check endpoint",
)
async def health_check():
    """Check application health and readiness."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        store_size=len(store.chunks),
        cache_enabled=settings.cache_enabled,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["monitoring"],
    summary="Application metrics",
)
async def get_metrics():
    """Get application metrics."""
    total = request_metrics["total_requests"]
    successful = request_metrics["successful_requests"]
    success_rate = (successful / total * 100) if total > 0 else 0.0

    return MetricsResponse(
        total_requests=total,
        successful_requests=successful,
        failed_requests=request_metrics["failed_requests"],
        success_rate=success_rate,
        cache_enabled=settings.cache_enabled,
    )


@app.post(
    "/ask",
    response_model=Answer,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid request or low confidence",
        },
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["rag"],
    summary="Ask a question about policies",
)
async def ask(q: Question) -> Answer:
    """Ask a question and get an answer with citations.

    The system will:
    1. Search for relevant policy documents
    2. Generate an answer using the LLM
    3. Validate citations and confidence
    4. Return answer or reject with error

    All answers must include citations in [DOC-ID] format.
    """
    start_time = time.time()
    request_metrics["total_requests"] += 1

    try:
        logger.info(f"Processing query: {q.query[:100]}")

        # Check cache
        cached_result = None
        cache_key = ""

        if settings.cache_enabled:
            cached_result, cache_key = cache.cached_get(q.query, settings.rag_top_k)

            if cached_result:
                logger.info("Returning cached result")
                request_metrics["successful_requests"] += 1
                process_time_ms = (time.time() - start_time) * 1000
                return Answer(
                    **cached_result, cached=True, process_time_ms=process_time_ms
                )

        # Search for relevant documents
        docs = await store.search(q.query, top_k=settings.rag_top_k)

        if not docs:
            logger.warning("No relevant documents found")
            request_metrics["failed_requests"] += 1
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No relevant policy documents found for your query",
            )

        # Generate answer with LLM
        llm = get_llm()
        context_text = "\n\n".join([f"[{d.id}] {d.text}" for d in docs])
        prompt = (
            f"Answer the following question using ONLY the provided context. "
            f"Include citations in [DOC-ID] format.\n\n"
            f"Question: {q.query}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Answer:"
        )

        draft = await llm.complete(prompt, docs)

        # Validate with guard
        decision = await approve_or_refuse(q.query, draft, docs)

        if not decision.allowed:
            logger.warning(f"Response rejected: {decision.reason}")
            request_metrics["failed_requests"] += 1
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=decision.reason
            )

        # Prepare response
        process_time_ms = (time.time() - start_time) * 1000
        result = {
            "answer": decision.answer,
            "sources": [d.id for d in docs],
            "confidence": decision.confidence,
            "citations_found": decision.citations_found,
        }

        # Cache result
        if settings.cache_enabled:
            cache.set(cache_key, result)

        request_metrics["successful_requests"] += 1
        logger.info(
            "Request successful",
            extra={
                "confidence": decision.confidence,
                "process_time_ms": process_time_ms,
            },
        )

        return Answer(**result, cached=False, process_time_ms=process_time_ms)

    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, e, {"query": q.query})
        request_metrics["failed_requests"] += 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred processing your request",
        ) from e


@app.get("/", tags=["info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Safe, auditable RAG system for enterprise policies",
        "docs_url": "/docs",
        "health_url": "/health",
    }
