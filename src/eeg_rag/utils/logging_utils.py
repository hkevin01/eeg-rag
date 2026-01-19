"""
Logging utilities for EEG-RAG system.

This module provides centralized logging configuration, performance monitoring,
and timing utilities. All time measurements are in seconds unless otherwise noted.

REQ-016: All time measurements must use consistent units (seconds)
REQ-017: Performance metrics must be logged for critical operations
REQ-018: Logging must not expose sensitive information
"""

import logging
import time
import functools
import traceback
import sys
from typing import Optional, Callable, Any
from pathlib import Path
from datetime import datetime
import json

# Time unit constants (all in seconds)
MILLISECOND = 0.001
SECOND = 1.0
MINUTE = 60.0
HOUR = 3600.0


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    include_timestamps: bool = True,
    include_module: bool = True,
) -> None:
    """
    Configure logging for the entire application.
    
    REQ-019: Logging must be configured at application startup
    REQ-020: Log format must include timestamps and module names
    REQ-021: Logs must be written to both console and file
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        include_timestamps: Include timestamps in log messages
        include_module: Include module names in log messages
    
    Raises:
        ValueError: If log_level is invalid
    """
    # Validate log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Build format string
    format_parts = []
    if include_timestamps:
        format_parts.append("%(asctime)s")
    format_parts.append("%(levelname)-8s")
    if include_module:
        format_parts.append("[%(name)s]")
    format_parts.append("%(message)s")
    
    log_format = " - ".join(format_parts)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[],
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logging.root.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(
                logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
            )
            logging.root.addHandler(file_handler)
            logging.info(f"Logging to file: {log_file}")
        except Exception as exc:
            logging.error(f"Failed to create log file {log_file}: {exc}")
    
    logging.info(f"Logging configured with level {log_level}")


def format_time(seconds: float, precision: int = 2) -> str:
    """
    Format time duration in human-readable format.
    
    REQ-022: Time durations must be displayed in human-readable format
    REQ-023: Time unit must be explicitly indicated
    
    All input and intermediate calculations are in seconds.
    
    Args:
        seconds: Time duration in seconds (can be fractional)
        precision: Number of decimal places for display
    
    Returns:
        Formatted time string with appropriate unit
    
    Examples:
        >>> format_time(0.00123)  # 1.23 milliseconds
        '1.23ms'
        >>> format_time(45.67)  # 45.67 seconds
        '45.67s'
        >>> format_time(125.5)  # 2 minutes 5.5 seconds
        '2m 5.50s'
        >>> format_time(7325)  # 2 hours 2 minutes 5 seconds
        '2h 2m 5s'
    """
    if seconds < 0:
        return f"Invalid: {seconds}s"
    
    # Less than 1 second: show milliseconds
    if seconds < 1.0:
        milliseconds = seconds / MILLISECOND
        return f"{milliseconds:.{precision}f}ms"
    
    # Less than 1 minute: show seconds
    if seconds < MINUTE:
        return f"{seconds:.{precision}f}s"
    
    # Less than 1 hour: show minutes and seconds
    if seconds < HOUR:
        minutes = int(seconds // MINUTE)
        remaining_seconds = seconds % MINUTE
        if remaining_seconds > 0.01:  # Only show seconds if significant
            return f"{minutes}m {remaining_seconds:.{precision}f}s"
        return f"{minutes}m"
    
    # 1 hour or more: show hours, minutes, seconds
    hours = int(seconds // HOUR)
    remaining_seconds = seconds % HOUR
    minutes = int(remaining_seconds // MINUTE)
    remaining_seconds = remaining_seconds % MINUTE
    
    parts = [f"{hours}h"]
    if minutes > 0:
        parts.append(f"{minutes}m")
    if remaining_seconds > 0.01:
        parts.append(f"{remaining_seconds:.0f}s")
    
    return " ".join(parts)


class PerformanceTimer:
    """
    Context manager for timing code blocks with automatic logging.
    
    REQ-024: Critical operations must be timed
    REQ-025: Timing results must be logged
    REQ-026: Timers must handle exceptions gracefully
    
    All timing measurements are in seconds.
    
    Example:
        >>> with PerformanceTimer("Loading embeddings"):
        ...     embeddings = load_large_file()
        INFO - Loading embeddings: completed in 2.35s
        
        >>> timer = PerformanceTimer("Processing", log_start=True)
        >>> with timer:
        ...     process_data()
        INFO - Processing: started
        INFO - Processing: completed in 1.23s
    """
    
    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        log_start: bool = False,
        log_completion: bool = True,
        warn_threshold_seconds: Optional[float] = None,
    ):
        """
        Initialize performance timer.
        
        Args:
            operation_name: Name of the operation being timed
            logger: Logger instance (uses root logger if None)
            log_start: Whether to log when operation starts
            log_completion: Whether to log when operation completes
            warn_threshold_seconds: Issue warning if operation exceeds this duration (in seconds)
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.log_start = log_start
        self.log_completion = log_completion
        self.warn_threshold_seconds = warn_threshold_seconds
        
        # Timing data (all in seconds)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_seconds: Optional[float] = None
    
    def __enter__(self) -> "PerformanceTimer":
        """Start timing when entering context."""
        self.start_time = time.perf_counter()  # High-resolution timer in seconds
        if self.log_start:
            self.logger.info(f"{self.operation_name}: started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Stop timing when exiting context.
        
        REQ-027: Timers must log duration even if exception occurs
        """
        self.end_time = time.perf_counter()
        self.duration_seconds = self.end_time - self.start_time
        
        if exc_type is not None:
            # Exception occurred
            self.logger.error(
                f"{self.operation_name}: failed after {format_time(self.duration_seconds)} "
                f"with {exc_type.__name__}: {exc_val}"
            )
        elif self.log_completion:
            # Normal completion
            formatted_duration = format_time(self.duration_seconds)
            
            # Check if duration exceeds warning threshold
            if (
                self.warn_threshold_seconds is not None
                and self.duration_seconds > self.warn_threshold_seconds
            ):
                self.logger.warning(
                    f"{self.operation_name}: completed in {formatted_duration} "
                    f"(exceeded threshold of {format_time(self.warn_threshold_seconds)})"
                )
            else:
                self.logger.info(
                    f"{self.operation_name}: completed in {formatted_duration}"
                )
    
    def get_duration_seconds(self) -> Optional[float]:
        """
        Get duration of timed operation in seconds.
        
        Returns:
            Duration in seconds, or None if timing not complete
        """
        return self.duration_seconds


def timed(
    operation_name: Optional[str] = None,
    log_args: bool = False,
    warn_threshold_seconds: Optional[float] = None,
) -> Callable:
    """
    Decorator for timing function execution.
    
    REQ-028: Critical functions must be timed automatically
    REQ-029: Function arguments must not be logged if they contain sensitive data
    
    Args:
        operation_name: Name for the operation (uses function name if None)
        log_args: Whether to log function arguments (careful with sensitive data!)
        warn_threshold_seconds: Issue warning if function exceeds this duration (in seconds)
    
    Returns:
        Decorated function with automatic timing
    
    Example:
        >>> @timed(warn_threshold_seconds=1.0)
        ... def slow_function(x: int) -> int:
        ...     time.sleep(1.5)
        ...     return x * 2
        >>> result = slow_function(5)
        WARNING - slow_function: completed in 1.50s (exceeded threshold of 1.00s)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal operation_name
            if operation_name is None:
                operation_name = func.__name__
            
            logger = logging.getLogger(func.__module__)
            
            # Log function call if requested
            if log_args:
                logger.debug(f"{operation_name} called with args={args}, kwargs={kwargs}")
            
            # Time the function execution
            with PerformanceTimer(
                operation_name, logger=logger, warn_threshold_seconds=warn_threshold_seconds
            ) as timer:
                result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    
    return decorator


def log_exception(
    logger: Optional[logging.Logger] = None,
    include_traceback: bool = True,
    reraise: bool = True,
) -> Callable:
    """
    Decorator for logging exceptions with full context.
    
    REQ-030: All exceptions must be logged with context
    REQ-031: Stack traces must be included for debugging
    REQ-032: Exceptions must be re-raised after logging (unless specified otherwise)
    
    Args:
        logger: Logger instance (uses root logger if None)
        include_traceback: Whether to include full stack trace
        reraise: Whether to re-raise exception after logging
    
    Returns:
        Decorated function with exception logging
    
    Example:
        >>> @log_exception(include_traceback=True, reraise=True)
        ... def risky_function(x: int) -> int:
        ...     return 10 / x
        >>> risky_function(0)
        ERROR - risky_function failed with ZeroDivisionError: division by zero
        ...
        ZeroDivisionError: division by zero
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            nonlocal logger
            if logger is None:
                logger = logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                # Build error message
                error_msg = f"{func.__name__} failed with {type(exc).__name__}: {exc}"
                
                if include_traceback:
                    logger.error(error_msg, exc_info=True)
                else:
                    logger.error(error_msg)
                
                # Re-raise if requested
                if reraise:
                    raise
                
                return None
        
        return wrapper
    
    return decorator


class PerformanceMonitor:
    """
    Collect and report performance metrics for the application.
    
    REQ-033: Performance metrics must be collected for analysis
    REQ-034: Metrics must be exportable for monitoring systems
    
    All timing data is stored in seconds.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: dict = {
            "operations": {},  # Operation name -> list of durations (seconds)
            "start_time": time.time(),  # Unix timestamp in seconds
        }
        self.logger = logging.getLogger(__name__)
    
    def record_operation(self, operation_name: str, duration_seconds: float) -> None:
        """
        Record a timed operation.
        
        Args:
            operation_name: Name of the operation
            duration_seconds: Duration in seconds
        """
        if operation_name not in self.metrics["operations"]:
            self.metrics["operations"][operation_name] = []
        
        self.metrics["operations"][operation_name].append(duration_seconds)
    
    def get_statistics(self, operation_name: str) -> Optional[dict]:
        """
        Get statistics for a specific operation.
        
        Returns:
            Dictionary with count, total, mean, min, max (all times in seconds)
        """
        if operation_name not in self.metrics["operations"]:
            return None
        
        durations = self.metrics["operations"][operation_name]
        return {
            "count": len(durations),
            "total_seconds": sum(durations),
            "mean_seconds": sum(durations) / len(durations),
            "min_seconds": min(durations),
            "max_seconds": max(durations),
        }
    
    def log_summary(self) -> None:
        """Log summary of all performance metrics."""
        self.logger.info("=== Performance Summary ===")
        
        total_runtime = time.time() - self.metrics["start_time"]
        self.logger.info(f"Total runtime: {format_time(total_runtime)}")
        
        for operation_name in self.metrics["operations"]:
            stats = self.get_statistics(operation_name)
            if stats:
                self.logger.info(
                    f"{operation_name}: {stats['count']} calls, "
                    f"avg {format_time(stats['mean_seconds'])}, "
                    f"total {format_time(stats['total_seconds'])}"
                )
    
    def export_metrics(self, filepath: Path) -> None:
        """
        Export metrics to JSON file.
        
        REQ-035: Metrics must be exportable in machine-readable format
        
        Args:
            filepath: Path to output JSON file
        """
        try:
            with open(filepath, "w", encoding="utf-8") as file:
                json.dump(self.metrics, file, indent=2)
            self.logger.info(f"Metrics exported to {filepath}")
        except Exception as exc:
            self.logger.error(f"Failed to export metrics: {exc}")


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _global_monitor


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (uses __name__ if None)
    
    Returns:
        Logger instance
    """
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            name = caller_frame.f_globals.get('__name__', 'eeg_rag')
        finally:
            del frame
    
    return logging.getLogger(name)
