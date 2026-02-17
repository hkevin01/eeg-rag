#!/usr/bin/env python3
"""
Centralized Error Handling Module for EEG-RAG

Provides standardized error types, error codes, and robust error handling
utilities for the entire application.

Requirements Implemented:
    - REQ-ERR-001: Descriptive error messages with context
    - REQ-ERR-002: Standardized error codes for programmatic handling
    - REQ-REL-002: Graceful error recovery mechanisms
    - REQ-SEC-001: Safe error messages (no sensitive data exposure)

Error Code Ranges:
    - 1000-1999: Input/Validation errors
    - 2000-2999: Retrieval/Search errors
    - 3000-3999: Citation/Verification errors
    - 4000-4999: Agent/Orchestration errors
    - 5000-5999: External API errors
    - 6000-6999: Database/Storage errors
    - 7000-7999: System/Infrastructure errors

Example Usage:
    >>> from eeg_rag.utils.error_handling import (
    ...     ValidationError, RetrievalError, safe_execute,
    ...     ErrorCode, handle_exception
    ... )
    >>> 
    >>> # Raise structured errors
    >>> if not query:
    ...     raise ValidationError(
    ...         ErrorCode.EMPTY_QUERY,
    ...         "Query cannot be empty",
    ...         context={"field": "query"}
    ...     )
    >>> 
    >>> # Safe execution with fallback
    >>> result = safe_execute(risky_function, default="fallback")
"""

import asyncio
import functools
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from .logging_utils import get_logger

logger = get_logger(__name__)

# Type variable for generic return types
T = TypeVar('T')


class ErrorCode(IntEnum):
    """
    Standardized error codes for programmatic error handling.
    
    REQ-ERR-002: All errors use these standard codes for consistent handling.
    
    Code Ranges:
        1000-1999: Input/Validation
        2000-2999: Retrieval/Search
        3000-3999: Citation/Verification
        4000-4999: Agent/Orchestration
        5000-5999: External API
        6000-6999: Database/Storage
        7000-7999: System/Infrastructure
    """
    # Success
    SUCCESS = 0
    
    # Input/Validation Errors (1000-1999)
    VALIDATION_ERROR = 1000
    EMPTY_QUERY = 1001
    QUERY_TOO_LONG = 1002
    INVALID_FORMAT = 1003
    INVALID_TYPE = 1004
    MISSING_REQUIRED_FIELD = 1005
    OUT_OF_RANGE = 1006
    MALFORMED_INPUT = 1007
    UNSUPPORTED_ENCODING = 1008
    INVALID_PMID_FORMAT = 1009
    
    # Retrieval/Search Errors (2000-2999)
    RETRIEVAL_ERROR = 2000
    NO_RESULTS_FOUND = 2001
    INDEX_NOT_FOUND = 2002
    EMBEDDING_FAILED = 2003
    SEARCH_TIMEOUT = 2004
    QUERY_EXPANSION_FAILED = 2005
    RERANKING_FAILED = 2006
    
    # Citation/Verification Errors (3000-3999)
    CITATION_ERROR = 3000
    PMID_NOT_FOUND = 3001
    PMID_RETRACTED = 3002
    ABSTRACT_UNAVAILABLE = 3003
    CLAIM_NOT_SUPPORTED = 3004
    VERIFICATION_TIMEOUT = 3005
    HALLUCINATION_DETECTED = 3006
    
    # Agent/Orchestration Errors (4000-4999)
    AGENT_ERROR = 4000
    AGENT_TIMEOUT = 4001
    AGENT_NOT_AVAILABLE = 4002
    ORCHESTRATION_FAILED = 4003
    ROUTING_ERROR = 4004
    AGGREGATION_FAILED = 4005
    AGENT_RESPONSE_INVALID = 4006
    
    # External API Errors (5000-5999)
    API_ERROR = 5000
    PUBMED_API_ERROR = 5001
    SEMANTIC_SCHOLAR_ERROR = 5002
    LLM_API_ERROR = 5003
    RATE_LIMITED = 5004
    API_TIMEOUT = 5005
    API_AUTHENTICATION_FAILED = 5006
    API_QUOTA_EXCEEDED = 5007
    
    # Database/Storage Errors (6000-6999)
    DATABASE_ERROR = 6000
    CONNECTION_FAILED = 6001
    QUERY_FAILED = 6002
    TRANSACTION_FAILED = 6003
    RECORD_NOT_FOUND = 6004
    DUPLICATE_RECORD = 6005
    STORAGE_FULL = 6006
    CACHE_ERROR = 6007
    
    # System/Infrastructure Errors (7000-7999)
    SYSTEM_ERROR = 7000
    OUT_OF_MEMORY = 7001
    DISK_FULL = 7002
    CONFIGURATION_ERROR = 7003
    DEPENDENCY_MISSING = 7004
    INITIALIZATION_FAILED = 7005
    SHUTDOWN_ERROR = 7006
    RESOURCE_EXHAUSTED = 7007


@dataclass
class ErrorContext:
    """
    Rich context information for error tracking and debugging.
    
    REQ-ERR-001: Provides detailed context for error diagnosis.
    
    Attributes:
        operation: Name of the operation that failed
        component: Component/module where error occurred
        input_summary: Sanitized summary of input (no sensitive data)
        stack_trace: Full stack trace if available
        timestamp: When the error occurred
        correlation_id: Request/session correlation ID
        additional_data: Additional contextual data
    """
    operation: str
    component: str = "unknown"
    input_summary: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    correlation_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation,
            'component': self.component,
            'input_summary': self.input_summary,
            'has_stack_trace': self.stack_trace is not None,
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'additional_data': self.additional_data,
        }


class EEGRAGError(Exception):
    """
    Base exception for all EEG-RAG errors.
    
    REQ-ERR-001: All errors include descriptive messages and context.
    REQ-ERR-002: All errors include standardized error codes.
    REQ-SEC-001: Error messages are safe (no sensitive data).
    
    Attributes:
        code: Standardized error code
        message: Human-readable error message
        context: Additional context for debugging
        recoverable: Whether the error can be recovered from
        user_message: Safe message to show to users
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        user_message: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize EEG-RAG error.
        
        Args:
            code: Standardized error code
            message: Detailed error message (for logging)
            context: Additional context for debugging
            recoverable: Whether the error can be recovered from
            user_message: Safe message for user display
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}
        self.recoverable = recoverable
        self.user_message = user_message or self._default_user_message()
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def _default_user_message(self) -> str:
        """Generate safe default user message based on error code."""
        # REQ-SEC-001: Safe messages that don't expose system details
        code_messages = {
            ErrorCode.EMPTY_QUERY: "Please enter a query to search.",
            ErrorCode.QUERY_TOO_LONG: "Your query is too long. Please shorten it.",
            ErrorCode.NO_RESULTS_FOUND: "No results found. Try different search terms.",
            ErrorCode.PMID_NOT_FOUND: "The referenced citation could not be found.",
            ErrorCode.SEARCH_TIMEOUT: "The search took too long. Please try again.",
            ErrorCode.RATE_LIMITED: "Too many requests. Please wait a moment.",
            ErrorCode.API_TIMEOUT: "The service is temporarily slow. Please try again.",
        }
        return code_messages.get(
            self.code, 
            "An error occurred. Please try again or contact support."
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for serialization.
        
        Returns:
            Dictionary with error details (safe for logging/API)
        """
        return {
            'error_code': self.code.value,
            'error_name': self.code.name,
            'message': self.message,
            'user_message': self.user_message,
            'recoverable': self.recoverable,
            'timestamp': self.timestamp,
            'context': {
                k: str(v)[:200] for k, v in self.context.items()
            },  # Truncate context values
        }
    
    def __str__(self) -> str:
        """Human-readable error representation."""
        return f"[{self.code.name}] {self.message}"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"{self.__class__.__name__}("
            f"code={self.code.name}, "
            f"message='{self.message[:50]}...', "
            f"recoverable={self.recoverable})"
        )


# Specialized exception classes for different error categories

class ValidationError(EEGRAGError):
    """
    Input validation errors.
    
    REQ-FUNC-002: Query validation with descriptive errors.
    """
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        message: str = "Validation failed",
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if field:
            context['field'] = field
        if value is not None:
            # Sanitize value for context
            context['value_type'] = type(value).__name__
            context['value_preview'] = str(value)[:50] if value else None
        super().__init__(code, message, context=context, **kwargs)
        self.field = field


class RetrievalError(EEGRAGError):
    """
    Document retrieval and search errors.
    
    REQ-FUNC-010: Hybrid retrieval error handling.
    """
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.RETRIEVAL_ERROR,
        message: str = "Retrieval failed",
        query: Optional[str] = None,
        source: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if query:
            context['query_preview'] = query[:100]
        if source:
            context['source'] = source
        super().__init__(code, message, context=context, **kwargs)


class CitationError(EEGRAGError):
    """
    Citation verification errors.
    
    REQ-FUNC-020: PMID validation and verification errors.
    """
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.CITATION_ERROR,
        message: str = "Citation verification failed",
        pmid: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if pmid:
            context['pmid'] = pmid
        super().__init__(code, message, context=context, **kwargs)
        self.pmid = pmid


class AgentError(EEGRAGError):
    """
    Agent and orchestration errors.
    
    REQ-FUNC-030: Multi-agent system error handling.
    """
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.AGENT_ERROR,
        message: str = "Agent operation failed",
        agent_name: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if agent_name:
            context['agent'] = agent_name
        super().__init__(code, message, context=context, **kwargs)
        self.agent_name = agent_name


class APIError(EEGRAGError):
    """
    External API errors.
    
    REQ-INT-001: External API integration error handling.
    """
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.API_ERROR,
        message: str = "API request failed",
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if api_name:
            context['api'] = api_name
        if status_code:
            context['status_code'] = status_code
        super().__init__(code, message, context=context, **kwargs)
        self.api_name = api_name
        self.status_code = status_code


class DatabaseError(EEGRAGError):
    """
    Database and storage errors.
    
    REQ-DAT-001: Persistence error handling.
    """
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.DATABASE_ERROR,
        message: str = "Database operation failed",
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.pop('context', {})
        if operation:
            context['operation'] = operation
        super().__init__(code, message, context=context, **kwargs)


class SystemError(EEGRAGError):
    """
    System and infrastructure errors.
    
    REQ-REL-001: System reliability error handling.
    """
    
    def __init__(
        self,
        code: ErrorCode = ErrorCode.SYSTEM_ERROR,
        message: str = "System error occurred",
        **kwargs
    ):
        super().__init__(code, message, recoverable=False, **kwargs)


# Error handling utilities

def handle_exception(
    exception: Exception,
    component: str = "unknown",
    operation: str = "unknown",
    log_level: str = "error",
    reraise: bool = True,
    default_code: ErrorCode = ErrorCode.SYSTEM_ERROR
) -> Optional[EEGRAGError]:
    """
    Standardized exception handling with logging.
    
    REQ-ERR-001: Consistent error handling across the application.
    
    Args:
        exception: The exception to handle
        component: Component where the error occurred
        operation: Operation that was being performed
        log_level: Logging level (debug, info, warning, error)
        reraise: Whether to re-raise as EEGRAGError
        default_code: Error code to use if exception is not EEGRAGError
        
    Returns:
        EEGRAGError instance if not re-raising
        
    Raises:
        EEGRAGError: If reraise is True
    """
    # Already an EEGRAGError, just log and optionally reraise
    if isinstance(exception, EEGRAGError):
        eeg_error = exception
    else:
        # Wrap in EEGRAGError
        eeg_error = EEGRAGError(
            code=default_code,
            message=str(exception),
            context={
                'component': component,
                'operation': operation,
                'exception_type': type(exception).__name__,
            },
            cause=exception
        )
    
    # Log with appropriate level
    log_func = getattr(logger, log_level.lower(), logger.error)
    log_func(
        f"Error in {component}.{operation}: "
        f"[{eeg_error.code.name}] {eeg_error.message}",
        exc_info=True
    )
    
    if reraise:
        raise eeg_error
    return eeg_error


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    log_errors: bool = True,
    **kwargs
) -> Optional[T]:
    """
    Execute a function with automatic error handling.
    
    REQ-REL-002: Safe execution with fallback values.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default: Default value to return on error
        on_error: Optional callback for error handling
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default value on error
        
    Example:
        >>> result = safe_execute(risky_function, arg1, default="fallback")
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(
                f"safe_execute caught error in {func.__name__}: {e}",
                exc_info=True
            )
        if on_error:
            on_error(e)
        return default


async def safe_execute_async(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    on_error: Optional[Callable[[Exception], None]] = None,
    log_errors: bool = True,
    **kwargs
) -> Optional[T]:
    """
    Execute an async function with automatic error handling.
    
    REQ-REL-002: Safe async execution with fallback values.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        default: Default value to return on error
        on_error: Optional callback for error handling
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default value on error
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(
                f"safe_execute_async caught error in {func.__name__}: {e}",
                exc_info=True
            )
        if on_error:
            on_error(e)
        return default


def with_error_handling(
    error_code: ErrorCode = ErrorCode.SYSTEM_ERROR,
    component: str = "unknown",
    log_errors: bool = True,
    reraise: bool = True
) -> Callable:
    """
    Decorator for standardized error handling.
    
    REQ-ERR-001: Consistent error handling decorator.
    
    Args:
        error_code: Default error code for unhandled exceptions
        component: Component name for logging
        log_errors: Whether to log errors
        reraise: Whether to re-raise as EEGRAGError
        
    Returns:
        Decorated function
        
    Example:
        >>> @with_error_handling(ErrorCode.RETRIEVAL_ERROR, "retriever")
        ... def search(query: str) -> List[Document]:
        ...     ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except EEGRAGError:
                raise
            except Exception as e:
                handle_exception(
                    e,
                    component=component,
                    operation=func.__name__,
                    log_level="error" if log_errors else "debug",
                    reraise=reraise,
                    default_code=error_code
                )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except EEGRAGError:
                raise
            except Exception as e:
                handle_exception(
                    e,
                    component=component,
                    operation=func.__name__,
                    log_level="error" if log_errors else "debug",
                    reraise=reraise,
                    default_code=error_code
                )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


def with_retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Decorator for automatic retry with exponential backoff.
    
    REQ-REL-002: Error recovery with retry logic.
    
    Args:
        max_attempts: Maximum number of attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        retryable_exceptions: Tuple of exception types to retry
        on_retry: Optional callback on each retry (exception, attempt_number)
        
    Returns:
        Decorated function
        
    Example:
        >>> @with_retry(max_attempts=3, delay_seconds=1.0)
        ... def fetch_data() -> dict:
        ...     ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = delay_seconds
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}"
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        
                        import time
                        time.sleep(delay)
                        delay *= backoff_multiplier
            
            # All retries exhausted
            raise EEGRAGError(
                code=ErrorCode.SYSTEM_ERROR,
                message=f"All {max_attempts} retry attempts failed for {func.__name__}",
                context={'last_error': str(last_exception)},
                cause=last_exception
            )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            delay = delay_seconds
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"Retry {attempt}/{max_attempts} for {func.__name__}: {e}"
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        
                        await asyncio.sleep(delay)
                        delay *= backoff_multiplier
            
            # All retries exhausted
            raise EEGRAGError(
                code=ErrorCode.SYSTEM_ERROR,
                message=f"All {max_attempts} retry attempts failed for {func.__name__}",
                context={'last_error': str(last_exception)},
                cause=last_exception
            )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


# Input validation utilities

def validate_not_empty(
    value: Any,
    field_name: str = "value"
) -> None:
    """
    Validate that a value is not empty.
    
    REQ-FUNC-002: Input validation utility.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        
    Raises:
        ValidationError: If value is None or empty
    """
    if value is None:
        raise ValidationError(
            code=ErrorCode.MISSING_REQUIRED_FIELD,
            message=f"{field_name} is required and cannot be None",
            field=field_name
        )
    
    if isinstance(value, str) and not value.strip():
        raise ValidationError(
            code=ErrorCode.EMPTY_QUERY,
            message=f"{field_name} cannot be empty",
            field=field_name,
            value=value
        )
    
    if isinstance(value, (list, dict, set)) and len(value) == 0:
        raise ValidationError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"{field_name} cannot be empty",
            field=field_name
        )


def validate_type(
    value: Any,
    expected_type: Union[Type, tuple],
    field_name: str = "value"
) -> None:
    """
    Validate that a value is of expected type.
    
    REQ-FUNC-002: Type validation utility.
    
    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        field_name: Name of the field for error messages
        
    Raises:
        ValidationError: If value is not of expected type
    """
    if not isinstance(value, expected_type):
        expected = (
            expected_type.__name__ 
            if isinstance(expected_type, type) 
            else str(expected_type)
        )
        raise ValidationError(
            code=ErrorCode.INVALID_TYPE,
            message=f"{field_name} must be {expected}, got {type(value).__name__}",
            field=field_name,
            value=value
        )


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    field_name: str = "value"
) -> None:
    """
    Validate that a numeric value is within range.
    
    REQ-FUNC-002: Range validation utility.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        field_name: Name of the field for error messages
        
    Raises:
        ValidationError: If value is out of range
    """
    if min_value is not None and value < min_value:
        raise ValidationError(
            code=ErrorCode.OUT_OF_RANGE,
            message=f"{field_name} must be >= {min_value}, got {value}",
            field=field_name,
            value=value,
            context={'min_value': min_value}
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            code=ErrorCode.OUT_OF_RANGE,
            message=f"{field_name} must be <= {max_value}, got {value}",
            field=field_name,
            value=value,
            context={'max_value': max_value}
        )


def validate_string_length(
    value: str,
    min_length: int = 0,
    max_length: Optional[int] = None,
    field_name: str = "value"
) -> None:
    """
    Validate string length constraints.
    
    REQ-FUNC-002: String length validation utility.
    
    Args:
        value: String to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        field_name: Name of the field for error messages
        
    Raises:
        ValidationError: If string length is out of bounds
    """
    actual_length = len(value)
    
    if actual_length < min_length:
        raise ValidationError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"{field_name} must be at least {min_length} characters, got {actual_length}",
            field=field_name,
            context={'min_length': min_length, 'actual_length': actual_length}
        )
    
    if max_length is not None and actual_length > max_length:
        raise ValidationError(
            code=ErrorCode.QUERY_TOO_LONG,
            message=f"{field_name} cannot exceed {max_length} characters, got {actual_length}",
            field=field_name,
            context={'max_length': max_length, 'actual_length': actual_length}
        )
