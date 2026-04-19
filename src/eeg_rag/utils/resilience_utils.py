"""
Resilience Utilities for EEG-RAG

Provides health monitoring, circuit breakers, rate limiting, graceful
degradation, and crash recovery mechanisms for production resilience.

Requirements Implemented:
- REQ-REL-001: Circuit breaker pattern
- REQ-REL-002: Health monitoring
- REQ-REL-004: Rate limiting
- REQ-REL-005: Graceful degradation
"""

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# REQ-REL-001: Circuit breaker states
# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.CircuitState
# Requirement  : `CircuitState` class shall be instantiable and expose the documented interface
# Purpose      : States for the circuit breaker pattern
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
# Verification : Instantiate CircuitState with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CircuitState(Enum):
    """States for the circuit breaker pattern."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.CircuitStats
# Requirement  : `CircuitStats` class shall be instantiable and expose the documented interface
# Purpose      : REQ-REL-001: Statistics for circuit breaker
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
# Verification : Instantiate CircuitStats with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class CircuitStats:
    """
    REQ-REL-001: Statistics for circuit breaker.
    
    Attributes:
        total_calls: Total number of calls
        failures: Number of failed calls
        successes: Number of successful calls
        rejected: Number of rejected calls (when open)
        last_failure: Timestamp of last failure
        last_success: Timestamp of last success
    """
    total_calls: int = 0
    failures: int = 0
    successes: int = 0
    rejected: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    state_changes: int = 0
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitStats.failure_rate
    # Requirement  : `failure_rate` shall calculate failure rate
    # Purpose      : Calculate failure rate
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : float
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
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failures / self.total_calls
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitStats.to_dict
    # Requirement  : `to_dict` shall serialize to dictionary
    # Purpose      : Serialize to dictionary
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
        """Serialize to dictionary."""
        return {
            "total_calls": self.total_calls,
            "failures": self.failures,
            "successes": self.successes,
            "rejected": self.rejected,
            "failure_rate": self.failure_rate,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "state_changes": self.state_changes,
        }


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.CircuitBreaker
# Requirement  : `CircuitBreaker` class shall be instantiable and expose the documented interface
# Purpose      : REQ-REL-001: Implements the circuit breaker pattern
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
# Verification : Instantiate CircuitBreaker with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """
    REQ-REL-001: Implements the circuit breaker pattern.
    
    Prevents cascading failures by failing fast when a service is unhealthy.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker.__init__
    # Requirement  : `__init__` shall initialize circuit breaker
    # Purpose      : Initialize circuit breaker
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str; failure_threshold: int (default=5); recovery_timeout: float (default=30.0); half_open_max_calls: int (default=3); monitored_exceptions: tuple (default=(Exception,))
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
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        monitored_exceptions: tuple = (Exception,),
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Identifier for this circuit breaker
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds to wait before half-open
            half_open_max_calls: Max test calls in half-open state
            monitored_exceptions: Exceptions that count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.monitored_exceptions = monitored_exceptions
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._stats = CircuitStats()
        self._lock = threading.Lock()
        self._state_change_callbacks: List[Callable] = []
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker.state
    # Requirement  : `state` shall get current circuit state, checking for timeout
    # Purpose      : Get current circuit state, checking for timeout
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : CircuitState
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
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker._should_attempt_reset
    # Requirement  : `_should_attempt_reset` shall check if enough time has passed to try recovery
    # Purpose      : Check if enough time has passed to try recovery
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : bool
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
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try recovery."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.recovery_timeout
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker._transition_to
    # Requirement  : `_transition_to` shall transition to a new state
    # Purpose      : Transition to a new state
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : new_state: CircuitState
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
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_changes += 1
        
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
        
        logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}")
        
        for callback in self._state_change_callbacks:
            try:
                callback(self.name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker.call
    # Requirement  : `call` shall execute function through circuit breaker
    # Purpose      : Execute function through circuit breaker
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : func: Callable[..., T]; *args; **kwargs
    # Outputs      : T
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
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        current_state = self.state  # Triggers state check
        
        with self._lock:
            self._stats.total_calls += 1
            
            if current_state == CircuitState.OPEN:
                self._stats.rejected += 1
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is open"
                )
            
            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._stats.rejected += 1
                    raise CircuitOpenError(
                        f"Circuit breaker '{self.name}' half-open limit reached"
                    )
                self._half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.monitored_exceptions as e:
            self._on_failure(e)
            raise
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker._on_success
    # Requirement  : `_on_success` shall handle successful call
    # Purpose      : Handle successful call
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self._stats.successes += 1
            self._stats.last_success = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.CLOSED)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker._on_failure
    # Requirement  : `_on_failure` shall handle failed call
    # Purpose      : Handle failed call
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : exception: Exception
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
    def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._stats.failures += 1
            self._stats.last_failure = datetime.now()
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker.reset
    # Requirement  : `reset` shall manually reset the circuit breaker
    # Purpose      : Manually reset the circuit breaker
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._last_failure_time = None
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker.on_state_change
    # Requirement  : `on_state_change` shall register a callback for state changes
    # Purpose      : Register a callback for state changes
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : callback: Callable
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
    def on_state_change(self, callback: Callable) -> None:
        """Register a callback for state changes."""
        self._state_change_callbacks.append(callback)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.CircuitBreaker.stats
    # Requirement  : `stats` shall get circuit breaker statistics
    # Purpose      : Get circuit breaker statistics
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : CircuitStats
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
    @property
    def stats(self) -> CircuitStats:
        """Get circuit breaker statistics."""
        return self._stats


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.CircuitOpenError
# Requirement  : `CircuitOpenError` class shall be instantiable and expose the documented interface
# Purpose      : Raised when trying to call through an open circuit
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
# Verification : Instantiate CircuitOpenError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class CircuitOpenError(Exception):
    """Raised when trying to call through an open circuit."""
    pass


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.circuit_breaker
# Requirement  : `circuit_breaker` shall rEQ-REL-001: Decorator to apply circuit breaker pattern
# Purpose      : REQ-REL-001: Decorator to apply circuit breaker pattern
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : name: str; failure_threshold: int (default=5); recovery_timeout: float (default=30.0)
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
def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> Callable:
    """
    REQ-REL-001: Decorator to apply circuit breaker pattern.
    
    Args:
        name: Circuit breaker identifier
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before recovery attempt
        
    Returns:
        Decorated function
    """
    cb = CircuitBreaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
    _circuit_breakers[name] = cb
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.decorator
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
        # ID           : utils.resilience_utils.wrapper
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
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = cb
        return wrapper
    
    return decorator


# Global registry of circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.get_circuit_breaker
# Requirement  : `get_circuit_breaker` shall get a circuit breaker by name
# Purpose      : Get a circuit breaker by name
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : name: str
# Outputs      : Optional[CircuitBreaker]
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
def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get a circuit breaker by name."""
    return _circuit_breakers.get(name)


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.get_all_circuit_breakers
# Requirement  : `get_all_circuit_breakers` shall get all registered circuit breakers
# Purpose      : Get all registered circuit breakers
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : None
# Outputs      : Dict[str, CircuitBreaker]
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
def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    return _circuit_breakers.copy()


# REQ-REL-002: Health check system
# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.HealthStatus
# Requirement  : `HealthStatus` class shall be instantiable and expose the documented interface
# Purpose      : Health check statuses
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
# Verification : Instantiate HealthStatus with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class HealthStatus(Enum):
    """Health check statuses."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.HealthCheckResult
# Requirement  : `HealthCheckResult` class shall be instantiable and expose the documented interface
# Purpose      : REQ-REL-002: Result of a health check
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
# Verification : Instantiate HealthCheckResult with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class HealthCheckResult:
    """
    REQ-REL-002: Result of a health check.
    
    Attributes:
        name: Name of the health check
        status: Health status
        message: Optional status message
        duration_ms: Time taken for check
        details: Additional details
    """
    name: str
    status: HealthStatus
    message: Optional[str] = None
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.HealthCheckResult.to_dict
    # Requirement  : `to_dict` shall serialize to dictionary
    # Purpose      : Serialize to dictionary
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
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.HealthChecker
# Requirement  : `HealthChecker` class shall be instantiable and expose the documented interface
# Purpose      : REQ-REL-002: Manages health checks for system components
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
# Verification : Instantiate HealthChecker with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class HealthChecker:
    """
    REQ-REL-002: Manages health checks for system components.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.HealthChecker.__init__
    # Requirement  : `__init__` shall initialize health checker
    # Purpose      : Initialize health checker
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._lock = threading.Lock()
        self._last_results: Dict[str, HealthCheckResult] = {}
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.HealthChecker.register
    # Requirement  : `register` shall register a health check
    # Purpose      : Register a health check
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str; check_fn: Callable[[], HealthCheckResult]
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
    def register(
        self,
        name: str,
        check_fn: Callable[[], HealthCheckResult],
    ) -> None:
        """
        Register a health check.
        
        Args:
            name: Unique name for the check
            check_fn: Function that returns HealthCheckResult
        """
        with self._lock:
            self._checks[name] = check_fn
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.HealthChecker.unregister
    # Requirement  : `unregister` shall unregister a health check
    # Purpose      : Unregister a health check
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str
    # Outputs      : bool
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
    def unregister(self, name: str) -> bool:
        """
        Unregister a health check.
        
        Args:
            name: Name of check to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                return True
            return False
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.HealthChecker.check
    # Requirement  : `check` shall run a specific health check
    # Purpose      : Run a specific health check
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str
    # Outputs      : HealthCheckResult
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
    def check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            name: Name of check to run
            
        Returns:
            HealthCheckResult
        """
        with self._lock:
            check_fn = self._checks.get(name)
        
        if check_fn is None:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check '{name}' not found",
            )
        
        start_time = time.time()
        try:
            result = check_fn()
            result.duration_ms = (time.time() - start_time) * 1000
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
        
        with self._lock:
            self._last_results[name] = result
        
        return result
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.HealthChecker.check_all
    # Requirement  : `check_all` shall run all registered health checks
    # Purpose      : Run all registered health checks
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, HealthCheckResult]
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
    def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of check names to results
        """
        with self._lock:
            check_names = list(self._checks.keys())
        
        results = {}
        for name in check_names:
            results[name] = self.check(name)
        
        return results
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.HealthChecker.get_overall_status
    # Requirement  : `get_overall_status` shall get overall system health status
    # Purpose      : Get overall system health status
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : HealthStatus
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
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.
        
        Returns:
            HealthStatus based on all checks
        """
        results = self.check_all()
        
        if not results:
            return HealthStatus.HEALTHY
        
        statuses = [r.status for r in results.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.HealthChecker.get_last_results
    # Requirement  : `get_last_results` shall get cached results from last check
    # Purpose      : Get cached results from last check
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict[str, HealthCheckResult]
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
    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """Get cached results from last check."""
        with self._lock:
            return self._last_results.copy()


# Global health checker instance
_health_checker = HealthChecker()


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.register_health_check
# Requirement  : `register_health_check` shall register a health check with the global checker
# Purpose      : Register a health check with the global checker
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : name: str; check_fn: Callable[[], HealthCheckResult]
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
def register_health_check(
    name: str,
    check_fn: Callable[[], HealthCheckResult],
) -> None:
    """Register a health check with the global checker."""
    _health_checker.register(name, check_fn)


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.check_health
# Requirement  : `check_health` shall check system health
# Purpose      : Check system health
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : name: Optional[str] (default=None)
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
def check_health(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Check system health.
    
    Args:
        name: Optional specific check to run
        
    Returns:
        Health check results
    """
    if name:
        result = _health_checker.check(name)
        return result.to_dict()
    
    results = _health_checker.check_all()
    overall = _health_checker.get_overall_status()
    
    return {
        "status": overall.value,
        "checks": {k: v.to_dict() for k, v in results.items()},
        "timestamp": datetime.now().isoformat(),
    }


# REQ-REL-004: Rate limiting
# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.RateLimiter
# Requirement  : `RateLimiter` class shall be instantiable and expose the documented interface
# Purpose      : REQ-REL-004: Token bucket rate limiter
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
# Verification : Instantiate RateLimiter with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    REQ-REL-004: Token bucket rate limiter.
    
    Controls the rate of operations to prevent overload.
    """
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.RateLimiter.__init__
    # Requirement  : `__init__` shall initialize rate limiter
    # Purpose      : Initialize rate limiter
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : rate: float; capacity: Optional[float] (default=None); name: str (default='default')
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
        rate: float,
        capacity: Optional[float] = None,
        name: str = "default",
    ):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens added per second
            capacity: Maximum token capacity (defaults to rate)
            name: Identifier for this rate limiter
        """
        self.rate = rate
        self.capacity = capacity or rate
        self.name = name
        
        self._tokens = self.capacity
        self._last_update = time.time()
        self._lock = threading.Lock()
        
        # Statistics
        self._total_requests = 0
        self._accepted = 0
        self._rejected = 0
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.RateLimiter._refill
    # Requirement  : `_refill` shall refill tokens based on elapsed time
    # Purpose      : Refill tokens based on elapsed time
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        new_tokens = elapsed * self.rate
        self._tokens = min(self.capacity, self._tokens + new_tokens)
        self._last_update = now
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.RateLimiter.acquire
    # Requirement  : `acquire` shall try to acquire tokens
    # Purpose      : Try to acquire tokens
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : tokens: float (default=1.0)
    # Outputs      : bool
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
    def acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limited
        """
        with self._lock:
            self._refill()
            self._total_requests += 1
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._accepted += 1
                return True
            else:
                self._rejected += 1
                return False
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.RateLimiter.wait
    # Requirement  : `wait` shall wait until tokens are available
    # Purpose      : Wait until tokens are available
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : tokens: float (default=1.0); timeout: Optional[float] (default=None)
    # Outputs      : bool
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
    def wait(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum seconds to wait
            
        Returns:
            True if tokens acquired, False if timed out
        """
        start_time = time.time()
        
        while True:
            if self.acquire(tokens):
                return True
            
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            # Calculate wait time
            with self._lock:
                wait_time = (tokens - self._tokens) / self.rate
            
            time.sleep(min(wait_time, 0.1))
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.RateLimiter.async_wait
    # Requirement  : `async_wait` shall async version of wait
    # Purpose      : Async version of wait
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : tokens: float (default=1.0); timeout: Optional[float] (default=None)
    # Outputs      : bool
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
    async def async_wait(
        self,
        tokens: float = 1.0,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Async version of wait.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum seconds to wait
            
        Returns:
            True if tokens acquired, False if timed out
        """
        start_time = time.time()
        
        while True:
            if self.acquire(tokens):
                return True
            
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            with self._lock:
                wait_time = (tokens - self._tokens) / self.rate
            
            await asyncio.sleep(min(wait_time, 0.1))
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.RateLimiter.stats
    # Requirement  : `stats` shall get rate limiter statistics
    # Purpose      : Get rate limiter statistics
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
    def stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "name": self.name,
                "rate": self.rate,
                "capacity": self.capacity,
                "tokens_available": self._tokens,
                "total_requests": self._total_requests,
                "accepted": self._accepted,
                "rejected": self._rejected,
                "rejection_rate": self._rejected / max(1, self._total_requests),
            }


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.rate_limit
# Requirement  : `rate_limit` shall rEQ-REL-004: Decorator to apply rate limiting
# Purpose      : REQ-REL-004: Decorator to apply rate limiting
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : rate: float; capacity: Optional[float] (default=None); block: bool (default=True); timeout: Optional[float] (default=None)
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
def rate_limit(
    rate: float,
    capacity: Optional[float] = None,
    block: bool = True,
    timeout: Optional[float] = None,
) -> Callable:
    """
    REQ-REL-004: Decorator to apply rate limiting.
    
    Args:
        rate: Tokens per second
        capacity: Token bucket capacity
        block: Whether to block or fail immediately
        timeout: Max wait time if blocking
        
    Returns:
        Decorated function
    """
    limiter = RateLimiter(rate=rate, capacity=capacity)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.decorator
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
        # ID           : utils.resilience_utils.wrapper
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
        @wraps(func)
        def wrapper(*args, **kwargs):
            if block:
                if not limiter.wait(timeout=timeout):
                    raise RateLimitExceeded(f"Rate limit timeout for {func.__name__}")
            else:
                if not limiter.acquire():
                    raise RateLimitExceeded(f"Rate limit exceeded for {func.__name__}")
            
            return func(*args, **kwargs)
        
        wrapper.rate_limiter = limiter
        return wrapper
    
    return decorator


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.RateLimitExceeded
# Requirement  : `RateLimitExceeded` class shall be instantiable and expose the documented interface
# Purpose      : Raised when rate limit is exceeded
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
# Verification : Instantiate RateLimitExceeded with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


# REQ-REL-005: Graceful degradation
# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.DegradationLevel
# Requirement  : `DegradationLevel` class shall be instantiable and expose the documented interface
# Purpose      : REQ-REL-005: Degradation level configuration
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
# Verification : Instantiate DegradationLevel with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
@dataclass
class DegradationLevel:
    """
    REQ-REL-005: Degradation level configuration.
    
    Attributes:
        name: Level name (e.g., "normal", "reduced", "minimal")
        priority: Priority threshold (lower = more critical)
        enabled_features: Features enabled at this level
    """
    name: str
    priority: int
    enabled_features: Set[str] = field(default_factory=set)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.DegradationLevel.is_feature_enabled
    # Requirement  : `is_feature_enabled` shall check if a feature is enabled at this level
    # Purpose      : Check if a feature is enabled at this level
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : feature: str
    # Outputs      : bool
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
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled at this level."""
        return feature in self.enabled_features


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.GracefulDegradation
# Requirement  : `GracefulDegradation` class shall be instantiable and expose the documented interface
# Purpose      : REQ-REL-005: Manages graceful degradation of services
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
# Verification : Instantiate GracefulDegradation with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class GracefulDegradation:
    """
    REQ-REL-005: Manages graceful degradation of services.
    
    Allows the system to continue operating with reduced functionality
    when under stress or when components fail.
    """
    
    # Standard degradation levels
    NORMAL = DegradationLevel("normal", 0, set())
    REDUCED = DegradationLevel("reduced", 1, set())
    MINIMAL = DegradationLevel("minimal", 2, set())
    EMERGENCY = DegradationLevel("emergency", 3, set())
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.GracefulDegradation.__init__
    # Requirement  : `__init__` shall initialize degradation manager
    # Purpose      : Initialize degradation manager
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
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
    def __init__(self):
        """Initialize degradation manager."""
        self._current_level = self.NORMAL
        self._feature_priorities: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._level_change_callbacks: List[Callable] = []
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.GracefulDegradation.register_feature
    # Requirement  : `register_feature` shall register a feature with its priority
    # Purpose      : Register a feature with its priority
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : name: str; priority: int
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
    def register_feature(self, name: str, priority: int) -> None:
        """
        Register a feature with its priority.
        
        Args:
            name: Feature name
            priority: Priority (0 = critical, higher = less critical)
        """
        with self._lock:
            self._feature_priorities[name] = priority
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.GracefulDegradation.set_level
    # Requirement  : `set_level` shall set the degradation level
    # Purpose      : Set the degradation level
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : level: DegradationLevel
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
    def set_level(self, level: DegradationLevel) -> None:
        """
        Set the degradation level.
        
        Args:
            level: New degradation level
        """
        with self._lock:
            old_level = self._current_level
            self._current_level = level
            
            # Update enabled features based on priority
            level.enabled_features = {
                name for name, priority in self._feature_priorities.items()
                if priority <= level.priority
            }
            
            logger.warning(
                f"Degradation level changed: {old_level.name} -> {level.name}"
            )
            
            for callback in self._level_change_callbacks:
                try:
                    callback(old_level, level)
                except Exception as e:
                    logger.error(f"Degradation callback error: {e}")
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.GracefulDegradation.is_feature_enabled
    # Requirement  : `is_feature_enabled` shall check if a feature is enabled at current level
    # Purpose      : Check if a feature is enabled at current level
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : feature: str
    # Outputs      : bool
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
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled at current level.
        
        Args:
            feature: Feature name
            
        Returns:
            True if feature is enabled
        """
        with self._lock:
            priority = self._feature_priorities.get(feature, 0)
            return priority <= self._current_level.priority
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.GracefulDegradation.current_level
    # Requirement  : `current_level` shall get current degradation level
    # Purpose      : Get current degradation level
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : DegradationLevel
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
    @property
    def current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        with self._lock:
            return self._current_level
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.GracefulDegradation.on_level_change
    # Requirement  : `on_level_change` shall register a callback for level changes
    # Purpose      : Register a callback for level changes
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : callback: Callable
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
    def on_level_change(self, callback: Callable) -> None:
        """Register a callback for level changes."""
        self._level_change_callbacks.append(callback)
    
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.GracefulDegradation.status
    # Requirement  : `status` shall get degradation status
    # Purpose      : Get degradation status
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
    def status(self) -> Dict[str, Any]:
        """Get degradation status."""
        with self._lock:
            return {
                "level": self._current_level.name,
                "priority": self._current_level.priority,
                "enabled_features": list(self._current_level.enabled_features),
                "all_features": {
                    name: {
                        "priority": priority,
                        "enabled": priority <= self._current_level.priority,
                    }
                    for name, priority in self._feature_priorities.items()
                },
            }


# Global degradation manager
_degradation = GracefulDegradation()


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.degrade_to
# Requirement  : `degrade_to` shall set the global degradation level
# Purpose      : Set the global degradation level
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : level: DegradationLevel
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
def degrade_to(level: DegradationLevel) -> None:
    """Set the global degradation level."""
    _degradation.set_level(level)


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.is_feature_enabled
# Requirement  : `is_feature_enabled` shall check if a feature is enabled globally
# Purpose      : Check if a feature is enabled globally
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : feature: str
# Outputs      : bool
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
def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled globally."""
    return _degradation.is_feature_enabled(feature)


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.feature_gate
# Requirement  : `feature_gate` shall rEQ-REL-005: Decorator for feature-gated functions
# Purpose      : REQ-REL-005: Decorator for feature-gated functions
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : feature: str
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
def feature_gate(feature: str) -> Callable:
    """
    REQ-REL-005: Decorator for feature-gated functions.
    
    Args:
        feature: Feature name to check
        
    Returns:
        Decorated function
    """
    # ---------------------------------------------------------------------------
    # ID           : utils.resilience_utils.decorator
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
        # ID           : utils.resilience_utils.wrapper
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
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_feature_enabled(feature):
                raise FeatureDisabledError(
                    f"Feature '{feature}' is disabled at current degradation level"
                )
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# ---------------------------------------------------------------------------
# ID           : utils.resilience_utils.FeatureDisabledError
# Requirement  : `FeatureDisabledError` class shall be instantiable and expose the documented interface
# Purpose      : Raised when a disabled feature is accessed
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
# Verification : Instantiate FeatureDisabledError with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class FeatureDisabledError(Exception):
    """Raised when a disabled feature is accessed."""
    pass


# Convenience exports
__all__ = [
    "CircuitState",
    "CircuitStats",
    "CircuitBreaker",
    "CircuitOpenError",
    "circuit_breaker",
    "get_circuit_breaker",
    "get_all_circuit_breakers",
    "HealthStatus",
    "HealthCheckResult",
    "HealthChecker",
    "register_health_check",
    "check_health",
    "RateLimiter",
    "rate_limit",
    "RateLimitExceeded",
    "DegradationLevel",
    "GracefulDegradation",
    "degrade_to",
    "is_feature_enabled",
    "feature_gate",
    "FeatureDisabledError",
]
