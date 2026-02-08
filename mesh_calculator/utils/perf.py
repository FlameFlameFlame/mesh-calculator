"""
Performance timing context manager for pipeline stages.
"""
import time

import structlog

logger = structlog.get_logger(__name__)


class PerfTimer:
    """Context manager for timing pipeline stages."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start
        logger.info("stage_completed", stage=self.stage_name, elapsed_s=round(elapsed, 3))
        return False
