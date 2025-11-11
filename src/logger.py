"""Structured logging configuration."""

import logging
import sys
from typing import Any

from .config import get_settings


def setup_logger(name: str = __name__) -> logging.Logger:
    """Configure and return a structured logger."""
    settings = get_settings()

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with structured format
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, settings.log_level.upper()))

    # Structured format for production, readable for development
    if settings.is_production:
        formatter = logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
            '"message":"%(message)s","function":"%(funcName)s","line":%(lineno)d}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def log_request(logger: logging.Logger, method: str, path: str, **kwargs: Any) -> None:
    """Log HTTP request with context."""
    logger.info(f"Request: {method} {path}", extra=kwargs)


def log_error(
    logger: logging.Logger, error: Exception, context: dict[str, Any]
) -> None:
    """Log error with full context."""
    logger.error(
        f"Error: {type(error).__name__}: {str(error)}", extra=context, exc_info=True
    )
