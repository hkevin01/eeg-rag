"""Utility modules for EEG-RAG system.

Provides common functionality for:
- Configuration management
- Logging and timing utilities
- Common validation patterns
- Error handling standards
- Time unit conversions
"""

from .config import Config
from .logging_utils import (
    setup_logging,
    PerformanceTimer,
    timed,
    log_exception,
    get_performance_monitor,
    format_time,
    SECOND, MINUTE, HOUR, MILLISECOND
)
from .common_utils import (
    validate_non_empty_string,
    validate_positive_number,
    validate_range,
    standardize_time_unit,
    safe_divide,
    safe_get_nested,
    compute_content_hash,
    retry_with_backoff,
    handle_database_operation,
    ensure_directory_exists,
    format_error_message
)

__all__ = [
    "Config",
    "setup_logging",
    "PerformanceTimer",
    "timed",
    "log_exception",
    "get_performance_monitor",
    "format_time",
    "validate_non_empty_string",
    "validate_positive_number",
    "validate_range", 
    "standardize_time_unit",
    "safe_divide",
    "safe_get_nested",
    "compute_content_hash",
    "retry_with_backoff",
    "handle_database_operation",
    "ensure_directory_exists",
    "format_error_message",
    "SECOND", "MINUTE", "HOUR", "MILLISECOND"
]