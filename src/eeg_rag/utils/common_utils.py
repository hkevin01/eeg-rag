"""
Common utility functions for EEG-RAG system.

This module consolidates frequently used patterns across the codebase:
- Input validation with consistent error messages
- Time unit standardization (all in seconds)
- Safe data access operations
- Content hashing for deduplication
- Retry mechanisms for external operations
- Database operation wrappers
- Error message formatting

All time measurements are in seconds unless otherwise specified.

REQ-UTL-001: Standardize validation across all modules
REQ-UTL-002: Consistent time unit handling (seconds)
REQ-UTL-003: Safe data access with error handling
REQ-UTL-004: Retry mechanisms for external dependencies
REQ-UTL-005: Standardized error messages
"""

import logging
import hashlib
import time
import asyncio
import sqlite3
import psutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
from functools import wraps
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Type variable for generic functions
T = TypeVar('T')

# Time unit constants (all in seconds)
MILLISECOND = 0.001
SECOND = 1.0
MINUTE = 60.0
HOUR = 3600.0
DAY = 86400.0

# Standard error messages
ERROR_MESSAGES = {
    'empty_string': "Value cannot be empty or contain only whitespace",
    'negative_number': "Value must be positive, got {value}",
    'out_of_range': "Value {value} must be between {min_val} and {max_val}",
    'invalid_type': "Expected {expected_type}, got {actual_type}",
    'none_value': "Value cannot be None",
    'invalid_time_unit': "Invalid time unit '{unit}', expected one of: {valid_units}"
}


def validate_non_empty_string(
    value: Optional[str],
    field_name: str = "value",
    allow_none: bool = False
) -> str:
    """
    Validate that a string is not None, empty, or whitespace-only.
    
    REQ-UTL-001: Standardized string validation
    
    Args:
        value: String to validate
        field_name: Name of field for error messages
        allow_none: If True, returns empty string for None input
        
    Returns:
        Validated string
        
    Raises:
        ValueError: If string is invalid
        
    Example:
        >>> validate_non_empty_string("hello", "query")
        "hello"
        >>> validate_non_empty_string("", "query")
        ValueError: query cannot be empty or contain only whitespace
    """
    if value is None:
        if allow_none:
            return ""
        raise ValueError(f"{field_name} cannot be None")
    
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string, got {type(value).__name__}"
        )
    
    if not value.strip():
        raise ValueError(f"{field_name} {ERROR_MESSAGES['empty_string']}")
    
    return value.strip()


def validate_positive_number(
    value: Union[int, float],
    field_name: str = "value",
    allow_zero: bool = False,
    min_value: Optional[float] = None
) -> Union[int, float]:
    """
    Validate that a number is positive.
    
    REQ-UTL-001: Standardized numeric validation
    
    Args:
        value: Number to validate
        field_name: Name of field for error messages
        allow_zero: If True, zero is considered valid
        min_value: Optional minimum value (overrides allow_zero)
        
    Returns:
        Validated number
        
    Raises:
        ValueError: If number is invalid
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be None")
    
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a number, got {type(value).__name__}"
        )
    
    if min_value is not None:
        if value < min_value:
            raise ValueError(
                f"{field_name} must be >= {min_value}, got {value}"
            )
    elif allow_zero:
        if value < 0:
            raise ValueError(
                f"{field_name} must be >= 0, got {value}"
            )
    else:
        if value <= 0:
            raise ValueError(
                f"{field_name} {ERROR_MESSAGES['negative_number'].format(value=value)}"
            )
    
    return value


def validate_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    field_name: str = "value",
    inclusive: bool = True
) -> Union[int, float]:
    """
    Validate that a number is within a specified range.
    
    REQ-UTL-001: Range validation with clear error messages
    
    Args:
        value: Number to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of field for error messages
        inclusive: If True, endpoints are included in valid range
        
    Returns:
        Validated number
        
    Raises:
        ValueError: If number is out of range
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be None")
    
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a number, got {type(value).__name__}"
        )
    
    if inclusive:
        if not (min_val <= value <= max_val):
            raise ValueError(
                ERROR_MESSAGES['out_of_range'].format(
                    value=value, min_val=min_val, max_val=max_val
                )
            )
    else:
        if not (min_val < value < max_val):
            raise ValueError(
                f"{field_name} must be between {min_val} and {max_val} (exclusive), got {value}"
            )
    
    return value


def standardize_time_unit(
    time_value: float,
    from_unit: str,
    to_unit: str = "seconds"
) -> float:
    """
    Convert time values between different units, always returning seconds.
    
    REQ-UTL-002: All time measurements must use consistent units (seconds)
    
    Args:
        time_value: Time value to convert
        from_unit: Source unit (seconds, minutes, hours, milliseconds)
        to_unit: Target unit (always "seconds" for consistency)
        
    Returns:
        Time value in seconds
        
    Raises:
        ValueError: If unit is not recognized
        
    Example:
        >>> standardize_time_unit(1.5, "minutes")
        90.0
        >>> standardize_time_unit(500, "milliseconds")
        0.5
    """
    validate_positive_number(time_value, "time_value", allow_zero=True)
    
    # Conversion factors to seconds
    unit_factors = {
        "milliseconds": MILLISECOND,
        "ms": MILLISECOND,
        "seconds": SECOND,
        "s": SECOND,
        "minutes": MINUTE,
        "min": MINUTE,
        "m": MINUTE,
        "hours": HOUR,
        "h": HOUR,
        "hr": HOUR
    }
    
    from_unit_lower = from_unit.lower().strip()
    
    if from_unit_lower not in unit_factors:
        valid_units = ", ".join(sorted(unit_factors.keys()))
        raise ValueError(
            ERROR_MESSAGES['invalid_time_unit'].format(
                unit=from_unit, valid_units=valid_units
            )
        )
    
    # Convert to seconds
    seconds_value = time_value * unit_factors[from_unit_lower]
    
    # For consistency, always return seconds
    if to_unit.lower() not in ["seconds", "s"]:
        logging.warning(
            f"Time conversion target unit '{to_unit}' changed to 'seconds' for consistency"
        )
    
    return seconds_value


def safe_divide(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: Union[int, float] = 0.0,
    field_name: str = "division"
) -> float:
    """
    Perform safe division with handling for zero denominator.
    
    REQ-UTL-003: Safe mathematical operations
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Value to return if denominator is zero
        field_name: Name for error reporting
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        logging.warning(f"{field_name}: Division by zero, returning default value {default}")
        return float(default)
    
    return float(numerator) / float(denominator)


def safe_get_nested(
    data: Dict[str, Any],
    keys: List[str],
    default: Any = None
) -> Any:
    """
    Safely access nested dictionary values.
    
    REQ-UTL-003: Safe data access with error handling
    
    Args:
        data: Dictionary to access
        keys: List of keys for nested access
        default: Value to return if key path doesn't exist
        
    Returns:
        Value at nested key path or default
        
    Example:
        >>> data = {"config": {"database": {"host": "localhost"}}}
        >>> safe_get_nested(data, ["config", "database", "host"])
        "localhost"
        >>> safe_get_nested(data, ["config", "missing"], "default")
        "default"
    """
    if not isinstance(data, dict):
        return default
    
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current


def compute_content_hash(
    content: str,
    algorithm: str = "md5",
    prefix: Optional[str] = None
) -> str:
    """
    Compute hash of content for deduplication.
    
    REQ-UTL-004: Standardized content hashing
    
    Args:
        content: Content to hash
        algorithm: Hash algorithm (md5, sha256)
        prefix: Optional prefix for hash
        
    Returns:
        Hex digest of content hash
        
    Example:
        >>> compute_content_hash("hello world")
        "5d41402abc4b2a76b9719d911017c592"
    """
    validate_non_empty_string(content, "content")
    
    if algorithm.lower() == "md5":
        hasher = hashlib.md5()
    elif algorithm.lower() == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hasher.update(content.encode('utf-8'))
    hash_value = hasher.hexdigest()
    
    if prefix:
        return f"{prefix}_{hash_value[:12]}"
    
    return hash_value


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying operations with exponential backoff.
    
    REQ-UTL-004: Retry mechanisms for external dependencies
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each failure
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions that trigger retry
        
    Example:
        @retry_with_backoff(max_retries=3, initial_delay=0.5)
        def unreliable_api_call():
            # Code that might fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Final attempt failed, re-raise
                        raise
                    
                    # Log the retry
                    logger = logging.getLogger(func.__module__)
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            # This shouldn't be reached, but just in case
            if last_exception:
                raise last_exception
        
        # Handle async functions
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                delay = initial_delay
                
                for attempt in range(max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_retries:
                            raise
                        
                        logger = logging.getLogger(func.__module__)
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1} failed: {str(e)}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        
                        await asyncio.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                
                if last_exception:
                    raise last_exception
            
            return async_wrapper
        
        return wrapper
    return decorator


def handle_database_operation(
    operation: Callable,
    db_path: Path,
    operation_name: str = "database operation",
    logger: Optional[logging.Logger] = None
) -> Any:
    """
    Handle database operations with proper error handling and logging.
    
    REQ-UTL-005: Standardized database operation handling
    
    Args:
        operation: Function that performs database operation
        db_path: Path to database file
        operation_name: Name of operation for logging
        logger: Logger instance
        
    Returns:
        Result of operation
        
    Raises:
        sqlite3.Error: If database operation fails
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        result = operation()
        logger.debug(f"{operation_name} completed successfully")
        return result
    
    except sqlite3.Error as e:
        error_msg = f"{operation_name} failed: {str(e)}"
        logger.error(error_msg)
        raise sqlite3.Error(error_msg) from e
    
    except Exception as e:
        error_msg = f"{operation_name} failed with unexpected error: {str(e)}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e


def ensure_directory_exists(
    path: Path,
    create_parents: bool = True,
    logger: Optional[logging.Logger] = None
) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    REQ-UTL-005: Standardized file system operations
    
    Args:
        path: Directory path to ensure exists
        create_parents: Create parent directories if needed
        logger: Logger instance
        
    Returns:
        Path to directory
        
    Raises:
        OSError: If directory cannot be created
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        if not path.exists():
            path.mkdir(parents=create_parents, exist_ok=True)
            logger.debug(f"Created directory: {path}")
        elif not path.is_dir():
            raise OSError(f"Path exists but is not a directory: {path}")
        
        return path
    
    except OSError as e:
        error_msg = f"Failed to ensure directory exists: {path} - {str(e)}"
        logger.error(error_msg)
        raise OSError(error_msg) from e


def format_error_message(
    operation: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    include_traceback: bool = False
) -> str:
    """
    Format standardized error messages.
    
    REQ-UTL-005: Standardized error message formatting
    
    Args:
        operation: Name of operation that failed
        error: Exception that occurred
        context: Additional context information
        include_traceback: Include traceback in message
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    message_parts = [
        f"Operation '{operation}' failed",
        f"Error: {error_type}: {error_msg}"
    ]
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        message_parts.append(f"Context: {context_str}")
    
    if include_traceback:
        import traceback
        tb_str = traceback.format_exc()
        message_parts.append(f"Traceback:\n{tb_str}")
    
    return " | ".join(message_parts)


def format_duration_human_readable(seconds: float, precision: int = 2) -> str:
    """
    Format duration in human-readable format.
    
    REQ-UTL-002: Consistent time formatting
    
    Args:
        seconds: Duration in seconds
        precision: Decimal places for display
        
    Returns:
        Human-readable duration string
        
    Example:
        >>> format_duration_human_readable(90.5)
        "1m 30.50s"
        >>> format_duration_human_readable(3661.25, 1)
        "1h 1m 1.2s"
    """
    if seconds < MINUTE:
        return f"{seconds:.{precision}f}s"
    elif seconds < HOUR:
        minutes = int(seconds // MINUTE)
        remaining_seconds = seconds % MINUTE
        return f"{minutes}m {remaining_seconds:.{precision}f}s"
    else:
        hours = int(seconds // HOUR)
        remaining_minutes = int((seconds % HOUR) // MINUTE)
        remaining_seconds = seconds % MINUTE
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.{precision}f}s"


# ============================================================================
# System Health Monitoring
# ============================================================================

class SystemStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemHealth:
    """
    System health metrics and status
    
    REQ-UTIL-006: System health monitoring
    """
    status: SystemStatus
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    timestamp: datetime
    warnings: List[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "status": self.status.value,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_percent": self.disk_percent,
            "timestamp": self.timestamp.isoformat(),
            "warnings": self.warnings,
            "metrics": self.metrics
        }


def check_system_health(
    cpu_warning_threshold: float = 80.0,
    cpu_critical_threshold: float = 95.0,
    memory_warning_threshold: float = 85.0,
    memory_critical_threshold: float = 95.0,
    disk_warning_threshold: float = 90.0,
    disk_critical_threshold: float = 98.0
) -> SystemHealth:
    """
    Check current system health metrics
    
    Args:
        cpu_warning_threshold: CPU usage warning threshold (%)
        cpu_critical_threshold: CPU usage critical threshold (%)
        memory_warning_threshold: Memory usage warning threshold (%)
        memory_critical_threshold: Memory usage critical threshold (%)
        disk_warning_threshold: Disk usage warning threshold (%)
        disk_critical_threshold: Disk usage critical threshold (%)
    
    Returns:
        SystemHealth object with current metrics and status
        
    REQ-UTIL-006: Monitor system resources to prevent failures
    """
    warnings = []
    status = SystemStatus.HEALTHY
    
    try:
        # Get CPU usage (1 second average)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get disk usage for root partition
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        # Additional metrics
        metrics = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "load_average": getattr(psutil, 'getloadavg', lambda: (0, 0, 0))()
        }
        
        # Check thresholds and set status
        if (cpu_percent >= cpu_critical_threshold or 
            memory_percent >= memory_critical_threshold or 
            disk_percent >= disk_critical_threshold):
            status = SystemStatus.CRITICAL
            
        elif (cpu_percent >= cpu_warning_threshold or 
              memory_percent >= memory_warning_threshold or 
              disk_percent >= disk_warning_threshold):
            status = SystemStatus.WARNING
            
        # Generate warnings
        if cpu_percent >= cpu_warning_threshold:
            warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
        if memory_percent >= memory_warning_threshold:
            warnings.append(f"High memory usage: {memory_percent:.1f}%")
        if disk_percent >= disk_warning_threshold:
            warnings.append(f"High disk usage: {disk_percent:.1f}%")
            
    except Exception as e:
        # Fallback if system monitoring fails
        status = SystemStatus.UNKNOWN
        cpu_percent = memory_percent = disk_percent = -1
        warnings.append(f"Failed to collect system metrics: {str(e)}")
        metrics = {}
    
    return SystemHealth(
        status=status,
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        disk_percent=disk_percent,
        timestamp=datetime.now(),
        warnings=warnings,
        metrics=metrics
    )


class CircuitBreakerState(Enum):
    """Circuit breaker states for external service protection"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker implementation for external service resilience
    
    REQ-UTIL-007: Protect against external service failures
    """
    name: str
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize circuit breaker"""
        self.logger = logging.getLogger(f"eeg_rag.circuit_breaker.{self.name}")
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute (can be async)
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Original exception: If function fails
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker {self.name} is OPEN"
                )
        
        try:
            # Execute function (handle both sync and async)
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Success - reset failure count
            if self.failure_count > 0:
                self.logger.info(
                    f"Circuit breaker {self.name} success - resetting failure count"
                )
                self.failure_count = 0
                self.state = CircuitBreakerState.CLOSED
                
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            self.logger.warning(
                f"Circuit breaker {self.name} failure {self.failure_count}/{self.failure_threshold}: {str(e)}"
            )
            
            # Check if we should open the circuit
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.error(
                    f"Circuit breaker {self.name} is now OPEN after {self.failure_count} failures"
                )
            
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.timeout_seconds
    
    def reset(self) -> None:
        """Manually reset circuit breaker"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.logger.info(f"Circuit breaker {self.name} manually reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


def create_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """
    Factory function for creating circuit breakers
    
    Args:
        name: Circuit breaker name
        **kwargs: Circuit breaker configuration
        
    Returns:
        Configured CircuitBreaker instance
    """
    return CircuitBreaker(name=name, **kwargs)


# Export public interface
__all__ = [
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
    "format_duration_human_readable",
    "check_system_health",
    "SystemHealth",
    "SystemStatus",
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerOpenError",
    "create_circuit_breaker",
    "MILLISECOND", "SECOND", "MINUTE", "HOUR", "DAY"
]