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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.ErrorCode
# Requirement  : `ErrorCode` class shall be instantiable and expose the documented interface
# Purpose      : Standardized error codes for programmatic error handling
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ErrorCode with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.ErrorContext
# Requirement  : `ErrorContext` class shall be instantiable and expose the documented interface
# Purpose      : Rich context information for error tracking and debugging
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ErrorContext with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.ErrorContext.to_dict
    # Requirement  : `to_dict` shall convert to dictionary for serialization
    # Purpose      : Convert to dictionary for serialization
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.EEGRAGError
# Requirement  : `EEGRAGError` class shall be instantiable and expose the documented interface
# Purpose      : Base exception for all EEG-RAG errors
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate EEGRAGError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.EEGRAGError.__init__
    # Requirement  : `__init__` shall initialize EEG-RAG error
    # Purpose      : Initialize EEG-RAG error
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : code: ErrorCode; message: str; context: Optional[Dict[str, Any]] (default=None); recoverable: bool (default=True); user_message: Optional[str] (default=None); cause: Optional[Exception] (default=None)
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.EEGRAGError._default_user_message
    # Requirement  : `_default_user_message` shall generate safe default user message based on error code
    # Purpose      : Generate safe default user message based on error code
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.EEGRAGError.to_dict
    # Requirement  : `to_dict` shall convert error to dictionary for serialization
    # Purpose      : Convert error to dictionary for serialization
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, Any]
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.EEGRAGError.__str__
    # Requirement  : `__str__` shall human-readable error representation
    # Purpose      : Human-readable error representation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __str__(self) -> str:
        """Human-readable error representation."""
        return f"[{self.code.name}] {self.message}"
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.EEGRAGError.__repr__
    # Requirement  : `__repr__` shall debug representation
    # Purpose      : Debug representation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : str
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"{self.__class__.__name__}("
            f"code={self.code.name}, "
            f"message='{self.message[:50]}...', "
            f"recoverable={self.recoverable})"
        )


# Specialized exception classes for different error categories

# ---------------------------------------------------------------------------
# ID           : utils.error_handling.ValidationError
# Requirement  : `ValidationError` class shall be instantiable and expose the documented interface
# Purpose      : Input validation errors
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate ValidationError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class ValidationError(EEGRAGError):
    """
    Input validation errors.
    
    REQ-FUNC-002: Query validation with descriptive errors.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.ValidationError.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : code: ErrorCode (default=ErrorCode.VALIDATION_ERROR); message: str (default='Validation failed'); field: Optional[str] (default=None); value: Optional[Any] (default=None); **kwargs
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.RetrievalError
# Requirement  : `RetrievalError` class shall be instantiable and expose the documented interface
# Purpose      : Document retrieval and search errors
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate RetrievalError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class RetrievalError(EEGRAGError):
    """
    Document retrieval and search errors.
    
    REQ-FUNC-010: Hybrid retrieval error handling.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.RetrievalError.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : code: ErrorCode (default=ErrorCode.RETRIEVAL_ERROR); message: str (default='Retrieval failed'); query: Optional[str] (default=None); source: Optional[str] (default=None); **kwargs
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.CitationError
# Requirement  : `CitationError` class shall be instantiable and expose the documented interface
# Purpose      : Citation verification errors
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate CitationError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CitationError(EEGRAGError):
    """
    Citation verification errors.
    
    REQ-FUNC-020: PMID validation and verification errors.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.CitationError.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : code: ErrorCode (default=ErrorCode.CITATION_ERROR); message: str (default='Citation verification failed'); pmid: Optional[str] (default=None); **kwargs
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.AgentError
# Requirement  : `AgentError` class shall be instantiable and expose the documented interface
# Purpose      : Agent and orchestration errors
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate AgentError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class AgentError(EEGRAGError):
    """
    Agent and orchestration errors.
    
    REQ-FUNC-030: Multi-agent system error handling.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.AgentError.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : code: ErrorCode (default=ErrorCode.AGENT_ERROR); message: str (default='Agent operation failed'); agent_name: Optional[str] (default=None); **kwargs
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.APIError
# Requirement  : `APIError` class shall be instantiable and expose the documented interface
# Purpose      : External API errors
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate APIError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class APIError(EEGRAGError):
    """
    External API errors.
    
    REQ-INT-001: External API integration error handling.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.APIError.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : code: ErrorCode (default=ErrorCode.API_ERROR); message: str (default='API request failed'); api_name: Optional[str] (default=None); status_code: Optional[int] (default=None); **kwargs
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.DatabaseError
# Requirement  : `DatabaseError` class shall be instantiable and expose the documented interface
# Purpose      : Database and storage errors
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate DatabaseError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class DatabaseError(EEGRAGError):
    """
    Database and storage errors.
    
    REQ-DAT-001: Persistence error handling.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.DatabaseError.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : code: ErrorCode (default=ErrorCode.DATABASE_ERROR); message: str (default='Database operation failed'); operation: Optional[str] (default=None); **kwargs
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.SystemError
# Requirement  : `SystemError` class shall be instantiable and expose the documented interface
# Purpose      : System and infrastructure errors
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate SystemError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class SystemError(EEGRAGError):
    """
    System and infrastructure errors.
    
    REQ-REL-001: System reliability error handling.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.SystemError.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : code: ErrorCode (default=ErrorCode.SYSTEM_ERROR); message: str (default='System error occurred'); **kwargs
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        code: ErrorCode = ErrorCode.SYSTEM_ERROR,
        message: str = "System error occurred",
        **kwargs
    ):
        super().__init__(code, message, recoverable=False, **kwargs)


# Error handling utilities

# ---------------------------------------------------------------------------
# ID           : utils.error_handling.handle_exception
# Requirement  : `handle_exception` shall standardized exception handling with logging
# Purpose      : Standardized exception handling with logging
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : exception: Exception; component: str (default='unknown'); operation: str (default='unknown'); log_level: str (default='error'); reraise: bool (default=True); default_code: ErrorCode (default=ErrorCode.SYSTEM_ERROR)
# Outputs      : Optional[EEGRAGError]
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.safe_execute
# Requirement  : `safe_execute` shall execute a function with automatic error handling
# Purpose      : Execute a function with automatic error handling
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : func: Callable[..., T]; default: Optional[T]; on_error: Optional[Callable[[Exception], None]]; log_errors: bool; *args; **kwargs
# Outputs      : Optional[T]
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.safe_execute_async
# Requirement  : `safe_execute_async` shall execute an async function with automatic error handling
# Purpose      : Execute an async function with automatic error handling
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : func: Callable[..., T]; default: Optional[T]; on_error: Optional[Callable[[Exception], None]]; log_errors: bool; *args; **kwargs
# Outputs      : Optional[T]
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Must be awaited (async)
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.with_error_handling
# Requirement  : `with_error_handling` shall decorator for standardized error handling
# Purpose      : Decorator for standardized error handling
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : error_code: ErrorCode (default=ErrorCode.SYSTEM_ERROR); component: str (default='unknown'); log_errors: bool (default=True); reraise: bool (default=True)
# Outputs      : Callable
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.decorator
    # Requirement  : `decorator` shall execute as specified
    # Purpose      : Decorator
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : func: Callable
    # Outputs      : Callable
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def decorator(func: Callable) -> Callable:
        # ---------------------------------------------------------------------------
        # ID           : utils.error_handling.wrapper
        # Requirement  : `wrapper` shall execute as specified
        # Purpose      : Wrapper
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : *args; **kwargs
        # Outputs      : Implicitly None or see body
        # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
        # Postcond.    : Return value satisfies documented output type and range
        # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
        # Side Effects : May update instance state or perform I/O; see body
        # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
        # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
        # Constraints  : Synchronous — must not block event loop
        # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
        # References   : EEG-RAG system design specification; see module docstring
        # ---------------------------------------------------------------------------
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
        
        # ---------------------------------------------------------------------------
        # ID           : utils.error_handling.async_wrapper
        # Requirement  : `async_wrapper` shall execute as specified
        # Purpose      : Async wrapper
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : *args; **kwargs
        # Outputs      : Implicitly None or see body
        # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
        # Postcond.    : Return value satisfies documented output type and range
        # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
        # Side Effects : May update instance state or perform I/O; see body
        # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
        # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
        # Constraints  : Must be awaited (async)
        # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
        # References   : EEG-RAG system design specification; see module docstring
        # ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.with_retry
# Requirement  : `with_retry` shall decorator for automatic retry with exponential backoff
# Purpose      : Decorator for automatic retry with exponential backoff
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : max_attempts: int (default=3); delay_seconds: float (default=1.0); backoff_multiplier: float (default=2.0); retryable_exceptions: tuple (default=(Exception,)); on_retry: Optional[Callable[[Exception, int], None]] (default=None)
# Outputs      : Callable
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
    # ---------------------------------------------------------------------------
    # ID           : utils.error_handling.decorator
    # Requirement  : `decorator` shall execute as specified
    # Purpose      : Decorator
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : func: Callable
    # Outputs      : Callable
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def decorator(func: Callable) -> Callable:
        # ---------------------------------------------------------------------------
        # ID           : utils.error_handling.wrapper
        # Requirement  : `wrapper` shall execute as specified
        # Purpose      : Wrapper
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : *args; **kwargs
        # Outputs      : Implicitly None or see body
        # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
        # Postcond.    : Return value satisfies documented output type and range
        # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
        # Side Effects : May update instance state or perform I/O; see body
        # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
        # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
        # Constraints  : Synchronous — must not block event loop
        # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
        # References   : EEG-RAG system design specification; see module docstring
        # ---------------------------------------------------------------------------
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
        
        # ---------------------------------------------------------------------------
        # ID           : utils.error_handling.async_wrapper
        # Requirement  : `async_wrapper` shall execute as specified
        # Purpose      : Async wrapper
        # Rationale    : Implements domain-specific logic per system design; see referenced specs
        # Inputs       : *args; **kwargs
        # Outputs      : Implicitly None or see body
        # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
        # Postcond.    : Return value satisfies documented output type and range
        # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
        # Side Effects : May update instance state or perform I/O; see body
        # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
        # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
        # Constraints  : Must be awaited (async)
        # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
        # References   : EEG-RAG system design specification; see module docstring
        # ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# ID           : utils.error_handling.validate_not_empty
# Requirement  : `validate_not_empty` shall validate that a value is not empty
# Purpose      : Validate that a value is not empty
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : value: Any; field_name: str (default='value')
# Outputs      : None
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.validate_type
# Requirement  : `validate_type` shall validate that a value is of expected type
# Purpose      : Validate that a value is of expected type
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : value: Any; expected_type: Union[Type, tuple]; field_name: str (default='value')
# Outputs      : None
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.validate_range
# Requirement  : `validate_range` shall validate that a numeric value is within range
# Purpose      : Validate that a numeric value is within range
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : value: Union[int, float]; min_value: Optional[Union[int, float]] (default=None); max_value: Optional[Union[int, float]] (default=None); field_name: str (default='value')
# Outputs      : None
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ID           : utils.error_handling.validate_string_length
# Requirement  : `validate_string_length` shall validate string length constraints
# Purpose      : Validate string length constraints
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : value: str; min_length: int (default=0); max_length: Optional[int] (default=None); field_name: str (default='value')
# Outputs      : None
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
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
