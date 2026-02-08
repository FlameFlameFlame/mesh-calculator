"""
Structured logging configuration for mesh calculator.

Supports two output modes:
- Development: colored, human-readable console output (default)
- Production: JSON lines (enabled via LOG_FORMAT=json env var)

Log level controlled via --verbose/--quiet CLI flags or LOG_LEVEL env var.
"""
import logging
import os

import structlog


def setup_logging(verbose: bool = False, quiet: bool = False):
    """
    Configure structlog with stdlib integration.

    Args:
        verbose: Enable DEBUG level logging
        quiet: Enable WARNING level logging (suppresses INFO)
    """
    level = os.environ.get(
        "LOG_LEVEL",
        "DEBUG" if verbose else "WARNING" if quiet else "INFO",
    ).upper()
    json_mode = os.environ.get("LOG_FORMAT", "").lower() == "json"

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_mode:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger("mesh_calculator")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
